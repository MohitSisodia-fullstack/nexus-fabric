# Fabric Notebook: Silver Layer — Semantic Chunking + Azure OpenAI Embeddings
# notebook: 02_silver_chunk_embed.py
# Run in: Microsoft Fabric / Databricks PySpark environment

import re
import json
import time
import hashlib
import logging
from typing import Iterator
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, udf, lit, current_timestamp, sha2, concat_ws, explode
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    ArrayType, FloatType, TimestampType
)
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus-fabric-silver")

# ─────────────────────────────────────────────
# CONFIG — set via Fabric Environment or secrets
# ─────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT   = spark.conf.get("spark.azure.openai.endpoint")
AZURE_OPENAI_KEY        = spark.conf.get("spark.azure.openai.key")
EMBEDDING_MODEL         = "text-embedding-3-small"   # 1536 dims, cheapest
BRONZE_TABLE            = "nexus_fabric.bronze.raw_documents"
SILVER_TABLE            = "nexus_fabric.silver.chunked_embeddings"
CHUNK_SIZE              = 512    # tokens target per chunk
CHUNK_OVERLAP           = 64     # token overlap between adjacent chunks
BATCH_SIZE              = 16     # OpenAI API batch size

# ─────────────────────────────────────────────
# 1. SEMANTIC CHUNKER
#    Splits on sentence boundaries, respects paragraph structure,
#    keeps chunks near CHUNK_SIZE tokens without breaking mid-sentence.
# ─────────────────────────────────────────────

def approximate_token_count(text: str) -> int:
    """Fast approx: 1 token ≈ 4 chars (OpenAI rule of thumb)."""
    return len(text) // 4


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex — no NLTK dependency."""
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Builds chunks by accumulating sentences until the token budget is spent,
    then starts a new chunk with an overlap window from the tail of the previous one.
    Returns list of {chunk_text, chunk_index, token_count}.
    """
    sentences   = split_into_sentences(text)
    chunks      = []
    current     = []
    current_len = 0
    chunk_idx   = 0

    for sentence in sentences:
        sent_len = approximate_token_count(sentence)

        # If adding this sentence overflows, flush the current chunk
        if current_len + sent_len > chunk_size and current:
            chunk_text = " ".join(current)
            chunks.append({
                "chunk_text":  chunk_text,
                "chunk_index": chunk_idx,
                "token_count": current_len
            })
            chunk_idx += 1

            # Carry the overlap tail into the next chunk
            overlap_tokens = 0
            overlap_sents  = []
            for s in reversed(current):
                t = approximate_token_count(s)
                if overlap_tokens + t <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_tokens += t
                else:
                    break
            current     = overlap_sents + [sentence]
            current_len = overlap_tokens + sent_len
        else:
            current.append(sentence)
            current_len += sent_len

    # Flush remainder
    if current:
        chunks.append({
            "chunk_text":  " ".join(current),
            "chunk_index": chunk_idx,
            "token_count": current_len
        })

    return chunks


# ─────────────────────────────────────────────
# 2. EMBEDDING FUNCTION (batched, with retry)
# ─────────────────────────────────────────────

def get_embeddings_batch(texts: list[str], client: openai.AzureOpenAI) -> list[list[float]]:
    """
    Calls Azure OpenAI Embeddings API in one batch request.
    Retries up to 3 times on rate limit (429) with exponential back-off.
    """
    for attempt in range(3):
        try:
            response = client.embeddings.create(
                model = EMBEDDING_MODEL,
                input = texts
            )
            return [item.embedding for item in response.data]
        except openai.RateLimitError:
            wait = 2 ** attempt
            logger.warning(f"Rate limit hit, waiting {wait}s (attempt {attempt+1}/3)")
            time.sleep(wait)
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    raise RuntimeError("Embedding API failed after 3 retries")


# ─────────────────────────────────────────────
# 3. PySpark UDF — chunk text (pure Python, serialisable)
# ─────────────────────────────────────────────

CHUNK_SCHEMA = ArrayType(StructType([
    StructField("chunk_text",  StringType(),  False),
    StructField("chunk_index", IntegerType(), False),
    StructField("token_count", IntegerType(), False),
]))

@udf(returnType=CHUNK_SCHEMA)
def chunk_text_udf(text: str):
    if not text:
        return []
    return semantic_chunk(text, CHUNK_SIZE, CHUNK_OVERLAP)


# ─────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_silver_pipeline():
    spark = SparkSession.builder.appName("nexus-fabric-silver").getOrCreate()

    # ── 4.1  Load Bronze layer ───────────────
    logger.info("Reading Bronze table...")
    bronze_df = spark.table(BRONZE_TABLE).filter(col("processed") == False)
    logger.info(f"Unprocessed documents: {bronze_df.count()}")

    # ── 4.2  Explode into chunks ─────────────
    logger.info("Chunking documents...")
    chunked_df = (
        bronze_df
        .withColumn("chunks", chunk_text_udf(col("content")))
        .withColumn("chunk",  explode(col("chunks")))
        .select(
            col("document_id"),
            col("source_path"),
            col("document_type"),
            col("chunk.chunk_text").alias("chunk_text"),
            col("chunk.chunk_index").alias("chunk_index"),
            col("chunk.token_count").alias("token_count"),
            sha2(col("chunk.chunk_text"), 256).alias("chunk_id"),
            col("metadata"),
        )
    )

    # ── 4.3  Embed in batches using foreachPartition ─────────
    logger.info("Generating embeddings (batched)...")

    embed_schema = StructType([
        StructField("chunk_id",   StringType(),         False),
        StructField("embedding",  ArrayType(FloatType()), False),
        StructField("embedded_at", TimestampType(),      True),
    ])

    def embed_partition(rows):
        """Called once per Spark partition — creates one OpenAI client per partition."""
        client = openai.AzureOpenAI(
            api_key     = AZURE_OPENAI_KEY,
            api_version = "2024-02-01",
            azure_endpoint = AZURE_OPENAI_ENDPOINT,
        )
        batch_texts = []
        batch_ids   = []

        def flush(texts, ids):
            embeddings = get_embeddings_batch(texts, client)
            from datetime import datetime
            ts = datetime.utcnow()
            return [
                {"chunk_id": cid, "embedding": emb, "embedded_at": ts}
                for cid, emb in zip(ids, embeddings)
            ]

        results = []
        for row in rows:
            batch_texts.append(row.chunk_text)
            batch_ids.append(row.chunk_id)
            if len(batch_texts) == BATCH_SIZE:
                results.extend(flush(batch_texts, batch_ids))
                batch_texts, batch_ids = [], []

        if batch_texts:
            results.extend(flush(batch_texts, batch_ids))

        return iter(results)

    embedding_rdd = chunked_df.rdd.mapPartitions(embed_partition)
    embedding_df  = spark.createDataFrame(embedding_rdd, embed_schema)

    # ── 4.4  Join embeddings back ─────────────
    silver_df = chunked_df.join(embedding_df, on="chunk_id", how="inner")

    # ── 4.5  Write Silver Delta table ────────
    logger.info(f"Writing Silver table: {SILVER_TABLE}")
    (
        silver_df.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(SILVER_TABLE)
    )

    # ── 4.6  Mark Bronze rows as processed ───
    processed_ids = [row.document_id for row in bronze_df.select("document_id").collect()]
    spark.sql(f"""
        UPDATE {BRONZE_TABLE}
        SET processed = true, processed_at = current_timestamp()
        WHERE document_id IN ({','.join(f"'{i}'" for i in processed_ids)})
    """)

    logger.info(f"✅ Silver pipeline complete. Chunks written: {silver_df.count()}")


if __name__ == "__main__":
    run_silver_pipeline()
