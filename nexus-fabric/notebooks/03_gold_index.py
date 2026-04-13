# Fabric Notebook: Gold Layer — Push to Azure AI Search (Hybrid Index)
# notebook: 03_gold_index.py

import os
import json
import logging
import time
from typing import Iterator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticSearch, SemanticPrioritizedFields,
    SemanticField, SearchField, VectorSearchAlgorithmKind
)
from azure.core.credentials import AzureKeyCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus-fabric-gold")

SEARCH_ENDPOINT = spark.conf.get("spark.azure.search.endpoint")
SEARCH_KEY      = spark.conf.get("spark.azure.search.key")
INDEX_NAME      = "nexus-fabric-index"
SILVER_TABLE    = "nexus_fabric.silver.chunked_embeddings"
BATCH_SIZE      = 100   # Azure AI Search upload batch


# ── 1.  Create/update the Hybrid Search index ───────────────────────────────

def create_or_update_index():
    """
    Creates a search index with:
    - BM25 full-text field  (keyword search)
    - HNSW vector field     (semantic/vector search)
    - Semantic ranking config (Azure Cognitive semantic ranker)
    Combined via RRF (Reciprocal Rank Fusion) at query time.
    """
    idx_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))

    fields = [
        SimpleField(name="id",            type=SearchFieldDataType.String, key=True),
        SimpleField(name="document_id",   type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index",   type=SearchFieldDataType.Int32,  sortable=True),
        SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_path",   type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="chunk_text", type=SearchFieldDataType.String,
                        analyzer_name="en.microsoft"),   # BM25 full-text field

        # Vector field — 1536 dims for text-embedding-3-small
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="nexus-hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="nexus-hnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters={"m": 4, "efConstruction": 400, "efSearch": 500, "metric": "cosine"},
            )
        ],
        profiles=[VectorSearchProfile(name="nexus-hnsw-profile", algorithm_configuration_name="nexus-hnsw")],
    )

    semantic_config = SemanticConfiguration(
        name="nexus-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="chunk_text")]
        ),
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
    )

    idx_client.create_or_update_index(index)
    logger.info(f"✅ Index '{INDEX_NAME}' created/updated with hybrid search config.")


# ── 2.  Upload documents ─────────────────────────────────────────────────────

def upload_partition(rows: Iterator) -> Iterator:
    """
    Uploads a Spark partition to Azure AI Search in batches of BATCH_SIZE.
    Called via foreachPartition so one SearchClient per partition.
    """
    client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))
    batch  = []

    def flush(b):
        for attempt in range(3):
            try:
                result = client.upload_documents(documents=b)
                failed = [r for r in result if not r.succeeded]
                if failed:
                    logger.warning(f"{len(failed)} docs failed to index")
                return
            except Exception as e:
                logger.warning(f"Upload attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        logger.error("Upload failed after 3 retries")

    for row in rows:
        doc = {
            "id":            row.chunk_id,
            "document_id":   row.document_id,
            "chunk_index":   row.chunk_index,
            "document_type": row.document_type,
            "source_path":   row.source_path,
            "chunk_text":    row.chunk_text,
            "embedding":     row.embedding,
        }
        batch.append(doc)
        if len(batch) >= BATCH_SIZE:
            flush(batch)
            batch = []

    if batch:
        flush(batch)

    return iter([])   # foreachPartition needs an iterator return


def run_gold_pipeline():
    spark = SparkSession.builder.appName("nexus-fabric-gold").getOrCreate()

    create_or_update_index()

    logger.info("Loading Silver embeddings...")
    silver_df = spark.table(SILVER_TABLE).filter(col("embedding").isNotNull())

    logger.info(f"Indexing {silver_df.count()} chunks...")
    silver_df.foreachPartition(upload_partition)

    logger.info("✅ Gold pipeline complete — all chunks indexed in Azure AI Search.")


if __name__ == "__main__":
    run_gold_pipeline()
