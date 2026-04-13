# 🧠 Nexus-Fabric
### Enterprise Multi-Modal RAG Platform with Automated LLMOps & Governance

[![CI/CD](https://github.com/YOUR_USERNAME/nexus-fabric/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/nexus-fabric/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Azure](https://img.shields.io/badge/Azure-OpenAI%20%7C%20AI%20Search%20%7C%20Fabric-0078d4)](https://azure.microsoft.com)

---

## 🏗️ Architecture

```
SharePoint / SQL / PDFs
        │
        ▼  (Fabric Data Factory)
┌─── BRONZE ───────────────────────┐
│  Raw ingest → OneLake → Delta    │
│  PII scan (Presidio guardrail)   │
└──────────────────────────────────┘
        │
        ▼  (PySpark Fabric Notebook)
┌─── SILVER ───────────────────────┐
│  Text cleaning → Semantic chunks │
│  Azure OpenAI text-embedding-3   │
└──────────────────────────────────┘
        │
        ▼  (Azure AI Search)
┌─── GOLD ─────────────────────────┐
│  Hybrid index (Vector + BM25)    │
│  RRF reranking + semantic rank   │
└──────────────────────────────────┘
        │
        ▼
┌─── APP LAYER ────────────────────┐
│  FastAPI + Streamlit UI          │
│  Query → Retrieve → GPT-4o       │
└──────────────────────────────────┘
```

---

## 🚀 Live Demo Deployment (for Resume)

### Option A — Streamlit Cloud (FREE, easiest recruiter link)

1. Fork this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app"
3. Point to `app/streamlit_app.py`
4. Add secrets in Streamlit Cloud dashboard:
   ```
   AZURE_OPENAI_KEY = "your-key"
   AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com"
   AZURE_SEARCH_ENDPOINT = "https://your-search.search.windows.net"
   AZURE_SEARCH_KEY = "your-key"
   ```
5. Your live link: `https://your-app-name.streamlit.app`

### Option B — Azure Container Apps (production-grade)

```bash
# 1. Build and push Docker image
docker build -t nexus-fabric ./app
az acr login --name yourregistry
docker tag nexus-fabric yourregistry.azurecr.io/nexus-fabric:latest
docker push yourregistry.azurecr.io/nexus-fabric:latest

# 2. Deploy to Azure Container Apps
az containerapp create \
  --name nexus-fabric \
  --resource-group nexus-rg \
  --image yourregistry.azurecr.io/nexus-fabric:latest \
  --target-port 8501 \
  --ingress external
```

### Option C — Demo without Azure (for offline/cheap demos)

The app includes a `DEMO_MODE=true` flag that uses pre-embedded sample documents
so recruiters see a working RAG chat without needing Azure credentials.

```bash
cd app
pip install -r requirements.txt
DEMO_MODE=true streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
nexus-fabric/
├── notebooks/
│   ├── 01_bronze_ingest.py          # Fabric Notebook: Data Factory → OneLake
│   ├── 02_silver_chunk_embed.py     # Fabric Notebook: PySpark chunking + embeddings
│   └── 03_gold_index.py             # Push vectors to Azure AI Search
├── app/
│   ├── streamlit_app.py             # Recruiter-facing demo UI
│   ├── rag_engine.py                # Core RAG logic (retrieve + generate)
│   ├── guardrails.py                # PII detection with Presidio
│   ├── Dockerfile
│   └── requirements.txt
├── tests/
│   ├── test_chunking.py
│   ├── test_pii_guardrail.py
│   └── test_rag_engine.py
├── .github/
│   └── workflows/
│       └── ci.yml                   # CI/CD pipeline
└── README.md
```

---

## ⚙️ Prerequisites

| Service | Tier | Approx Cost |
|---|---|---|
| Microsoft Fabric | F2 trial (free 60 days) | Free |
| Azure OpenAI | Pay-per-token | ~$5 for demo |
| Azure AI Search | Free tier (3 indexes) | Free |
| Azure Container Apps | Consumption plan | ~$0–$5/month |

---

## 🔑 Environment Variables

```env
# Azure OpenAI
AZURE_OPENAI_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT_EMBEDDING=text-embedding-3-small
AZURE_OPENAI_DEPLOYMENT_CHAT=gpt-4o

# Azure AI Search
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_KEY=
AZURE_SEARCH_INDEX=nexus-fabric-index

# Demo mode (no Azure needed)
DEMO_MODE=false
```

---

## 🔒 Key Technical Highlights (for Interview)

- **Hybrid Search**: Combines vector similarity (cosine) with BM25 keyword search using Reciprocal Rank Fusion (RRF) — increases retrieval accuracy by ~30% vs vector-only
- **PII Guardrails**: Microsoft Presidio scans every chunk before it reaches the LLM — prevents leakage of names, emails, SSNs
- **Medallion Architecture**: Bronze → Silver → Gold ensures data lineage, reprocessing capability, and separation of raw vs enriched data
- **Semantic Chunking**: Chunks split on sentence boundaries + semantic similarity threshold, not fixed character counts
