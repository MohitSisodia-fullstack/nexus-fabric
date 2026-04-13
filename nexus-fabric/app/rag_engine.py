# app/rag_engine.py
# Core RAG Engine — Hybrid Search (Vector + BM25) + GPT-4o generation

import os
import logging
from dataclasses import dataclass
from typing import Optional
import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from guardrails import scan_before_llm

logger = logging.getLogger("nexus-rag")

# ── CONFIG ──────────────────────────────────────────────────────────────────
AZURE_OAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OAI_KEY        = os.getenv("AZURE_OPENAI_KEY", "")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDING", "text-embedding-3-small")
CHAT_DEPLOYMENT      = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT", "gpt-4o")

SEARCH_ENDPOINT      = os.getenv("AZURE_SEARCH_ENDPOINT", "")
SEARCH_KEY           = os.getenv("AZURE_SEARCH_KEY", "")
SEARCH_INDEX         = os.getenv("AZURE_SEARCH_INDEX", "nexus-fabric-index")

TOP_K                = 5      # chunks to retrieve
DEMO_MODE            = os.getenv("DEMO_MODE", "false").lower() == "true"

# ── DEMO DOCUMENTS (used when DEMO_MODE=true) ───────────────────────────────
DEMO_DOCS = [
    {"id": "d1", "chunk_text": "Nexus-Fabric uses a Medallion Architecture with Bronze, Silver, and Gold layers. Bronze ingests raw PDFs and SharePoint files. Silver applies PySpark transformations and generates embeddings. Gold serves the indexed vectors via Azure AI Search.", "source_path": "architecture_overview.pdf"},
    {"id": "d2", "chunk_text": "The PII Guardrail layer scans every chunk using Microsoft Presidio before sending data to the LLM. Entities like SSN, credit card numbers, and email addresses are either redacted or block the entire request.", "source_path": "security_design.pdf"},
    {"id": "d3", "chunk_text": "Hybrid Search combines BM25 keyword matching with vector similarity using cosine distance. Reciprocal Rank Fusion (RRF) merges the two ranked lists. This improves retrieval accuracy by approximately 30% compared to vector-only search.", "source_path": "retrieval_strategy.pdf"},
    {"id": "d4", "chunk_text": "The GitHub Actions CI/CD pipeline runs unit tests on chunking logic, PII detection, and RAG retrieval on every pull request. A passing build automatically deploys the Streamlit app to Azure Container Apps.", "source_path": "devops_guide.pdf"},
    {"id": "d5", "chunk_text": "Semantic chunking splits text at sentence boundaries, respecting paragraph structure. Each chunk targets 512 tokens with a 64-token overlap to preserve context across chunk boundaries.", "source_path": "chunking_design.pdf"},
]


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    query_sanitized: bool
    pii_found: list[str]
    retrieval_mode: str   # "hybrid" | "demo"


class RAGEngine:
    def __init__(self):
        if not DEMO_MODE:
            self._oai_client = openai.AzureOpenAI(
                api_key=AZURE_OAI_KEY,
                api_version="2024-02-01",
                azure_endpoint=AZURE_OAI_ENDPOINT,
            )
            self._search_client = SearchClient(
                SEARCH_ENDPOINT, SEARCH_INDEX, AzureKeyCredential(SEARCH_KEY)
            )
        logger.info(f"RAGEngine initialized. Demo mode: {DEMO_MODE}")

    # ── Query embedding ──────────────────────────────────────────────────────
    def _embed_query(self, query: str) -> list[float]:
        resp = self._oai_client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT, input=query
        )
        return resp.data[0].embedding

    # ── Hybrid retrieval ─────────────────────────────────────────────────────
    def _hybrid_search(self, query: str, query_vector: list[float]) -> list[dict]:
        """
        Hybrid search = BM25 full-text + HNSW vector search, fused via RRF.
        Azure AI Search handles the fusion natively with query_type="semantic".
        """
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=TOP_K * 2,   # over-fetch, let RRF rerank
            fields="embedding",
        )
        results = self._search_client.search(
            search_text=query,           # BM25 arm
            vector_queries=[vector_query],  # Vector arm
            query_type="semantic",
            semantic_configuration_name="nexus-semantic-config",
            top=TOP_K,
            select=["id", "chunk_text", "source_path", "document_type", "chunk_index"],
        )
        return [
            {
                "id":            r["id"],
                "chunk_text":    r["chunk_text"],
                "source_path":   r["source_path"],
                "document_type": r.get("document_type", ""),
                "score":         r["@search.score"],
            }
            for r in results
        ]

    # ── Demo retrieval (keyword only, no Azure) ──────────────────────────────
    def _demo_search(self, query: str) -> list[dict]:
        query_lower = query.lower()
        scored = []
        for doc in DEMO_DOCS:
            words   = set(query_lower.split())
            text_l  = doc["chunk_text"].lower()
            score   = sum(1 for w in words if w in text_l)
            scored.append({**doc, "score": score, "document_type": "demo"})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:TOP_K]

    # ── Generate answer ──────────────────────────────────────────────────────
    def _generate(self, query: str, chunks: list[dict]) -> str:
        context = "\n\n---\n\n".join(
            f"[Source: {c['source_path']}]\n{c['chunk_text']}" for c in chunks
        )
        system_prompt = (
            "You are Nexus-Fabric, an enterprise knowledge assistant. "
            "Answer the user's question using ONLY the context provided below. "
            "If the answer is not in the context, say so clearly. "
            "Always cite the source file name when referencing information.\n\n"
            f"CONTEXT:\n{context}"
        )
        if DEMO_MODE:
            # Simulate LLM response for offline demo
            return (
                f"Based on the Nexus-Fabric documentation:\n\n"
                + "\n\n".join(
                    f"**{c['source_path']}**: {c['chunk_text'][:200]}..."
                    for c in chunks[:2]
                )
            )

        resp = self._oai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        return resp.choices[0].message.content

    # ── Main entry point ─────────────────────────────────────────────────────
    def query(self, user_query: str) -> RAGResponse:
        """Full RAG pipeline: sanitize → embed → retrieve → generate."""

        # 1. PII guardrail on the user's query
        guard = scan_before_llm(user_query)
        if guard.blocked:
            return RAGResponse(
                answer=f"⚠️ Request blocked: {guard.block_reason}",
                sources=[], query_sanitized=True,
                pii_found=guard.pii_found, retrieval_mode="blocked"
            )
        safe_query = guard.sanitized_text

        # 2. Retrieve
        if DEMO_MODE:
            chunks = self._demo_search(safe_query)
            mode   = "demo"
        else:
            vec    = self._embed_query(safe_query)
            chunks = self._hybrid_search(safe_query, vec)
            mode   = "hybrid"

        # 3. Generate
        answer = self._generate(safe_query, chunks)

        return RAGResponse(
            answer=answer,
            sources=chunks,
            query_sanitized=bool(guard.pii_found),
            pii_found=guard.pii_found,
            retrieval_mode=mode,
        )


# Singleton
_engine: Optional[RAGEngine] = None

def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
