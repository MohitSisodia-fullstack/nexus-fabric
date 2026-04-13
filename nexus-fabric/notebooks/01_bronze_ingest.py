# Fabric Notebook: Bronze Layer — Ingest from SharePoint/SQL → OneLake
# notebook: 01_bronze_ingest.py

import os
import uuid
import hashlib
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, current_timestamp, input_file_name, sha2
)
from pyspark.sql.types import (
    StructType, StructField, StringType, BooleanType,
    TimestampType, MapType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus-fabric-bronze")

# ── CONFIG ──────────────────────────────────
ONELAKE_PATH     = "abfss://nexus@onelake.dfs.fabric.microsoft.com/NexusFabric.Lakehouse"
BRONZE_TABLE     = "nexus_fabric.bronze.raw_documents"
SOURCE_PDF_PATH  = f"{ONELAKE_PATH}/Files/raw/pdfs/"
SOURCE_SHAREPOINT_TABLE = "nexus_fabric.staging.sharepoint_docs"


def ingest_pdfs():
    """
    Reads PDFs from OneLake Files area (pre-loaded by Fabric Data Pipeline),
    extracts text with pdfplumber, and writes to Bronze Delta table.
    """
    spark = SparkSession.builder.appName("nexus-bronze-ingest").getOrCreate()

    # In Fabric, PDFs are first uploaded to OneLake Files via Data Factory.
    # We use a binary file reader then extract text per partition.
    pdf_binary_df = (
        spark.read
        .format("binaryFile")
        .option("pathGlobFilter", "*.pdf")
        .option("recursiveFileLookup", "true")
        .load(SOURCE_PDF_PATH)
    )

    def extract_pdf_text(binary_content: bytes) -> str:
        """Extract text from PDF binary using pdfplumber."""
        import io
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(binary_content)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n\n".join(pages)
        except Exception as e:
            logger.warning(f"PDF extraction failed: {e}")
            return ""

    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    extract_udf = udf(extract_pdf_text, StringType())

    bronze_schema = StructType([
        StructField("document_id",    StringType(),               False),
        StructField("source_path",    StringType(),               False),
        StructField("document_type",  StringType(),               True),
        StructField("content",        StringType(),               True),
        StructField("content_hash",   StringType(),               True),
        StructField("metadata",       MapType(StringType(), StringType()), True),
        StructField("ingested_at",    TimestampType(),            True),
        StructField("processed",      BooleanType(),              False),
    ])

    bronze_df = (
        pdf_binary_df
        .withColumn("content",       extract_udf(col("content")))
        .withColumn("document_id",   sha2(col("path"), 256))
        .withColumn("source_path",   col("path"))
        .withColumn("document_type", lit("pdf"))
        .withColumn("content_hash",  sha2(col("content"), 256))
        .withColumn("ingested_at",   current_timestamp())
        .withColumn("processed",     lit(False))
        .withColumn("metadata",      lit(None).cast(MapType(StringType(), StringType())))
        .select(
            "document_id", "source_path", "document_type",
            "content", "content_hash", "metadata", "ingested_at", "processed"
        )
    )

    # Deduplicate by content hash (avoid re-processing same file)
    try:
        existing = spark.table(BRONZE_TABLE).select("content_hash")
        bronze_df = bronze_df.join(existing, on="content_hash", how="left_anti")
    except Exception:
        pass  # Table doesn't exist yet on first run

    count = bronze_df.count()
    logger.info(f"New documents to ingest: {count}")

    if count > 0:
        (
            bronze_df.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(BRONZE_TABLE)
        )
        logger.info(f"✅ Bronze ingest complete. {count} documents written.")
    else:
        logger.info("No new documents found.")


def ingest_sharepoint():
    """
    SharePoint documents are synced by Fabric Data Pipeline into a staging Delta table.
    This function promotes them to the Bronze canonical table.
    """
    spark = SparkSession.builder.appName("nexus-bronze-sharepoint").getOrCreate()

    sharepoint_df = (
        spark.table(SOURCE_SHAREPOINT_TABLE)
        .withColumn("document_type", lit("sharepoint"))
        .withColumn("ingested_at",   current_timestamp())
        .withColumn("processed",     lit(False))
        .withColumnRenamed("file_path", "source_path")
        .withColumnRenamed("body_text", "content")
        .withColumn("document_id",   sha2(col("source_path"), 256))
        .withColumn("content_hash",  sha2(col("content"), 256))
    )

    sharepoint_df.write.format("delta").mode("append").saveAsTable(BRONZE_TABLE)
    logger.info("✅ SharePoint Bronze ingest complete.")


if __name__ == "__main__":
    ingest_pdfs()
    ingest_sharepoint()
