# Databricks notebook source
# MAGIC %md
# MAGIC # CE Document Master Creation Pipeline
# MAGIC ## This notebook processes CE documents for inference, extracting text and creating master tables with support for both single and dual commodity documents.

# COMMAND ----------

# MAGIC %md
# MAGIC ### DO NOT MODIFY THIS PART AND LEAVE AS IT IS
# MAGIC ### Fixed section to make MLops working

# COMMAND ----------

# DBTITLE 1,Set notebook extensions
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Install requirements
## MAGIC %pip install -r /Workspace/.../src/requirements.txt
## MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Set working directory
import os
import sys
from databricks.sdk.runtime import dbutils

# Set notebook working directory
notebook_path = "/Workspace/" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
os.chdir(notebook_path)

# Allow imports from src directory
sys.path.append("../")

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDITABLE SECTION
# MAGIC ### Here you can put your code, starting from imports

# COMMAND ----------

# DBTITLE 1,Import libraries
import logging
from pyspark.sql import SparkSession
from src.config.config import Config
import pyspark.sql.types as T
import pyspark.sql.functions as F
from src.utils.logger import get_logger
from src.master.master_creation import TextExtractor
from src.master.master_preprocessing import token_count_udf, filter_serving_rows_udf, remove_numbers
from src.utils.utils import get_campaign_dates, build_schema

# COMMAND ----------

# DBTITLE 1,Initialize logger and Spark session
# Initialize logging and Spark for CE processing
logger = get_logger()
spark: SparkSession = SparkSession.builder.getOrCreate()
logger_py4j = logging.getLogger("py4j")
logger_py4j.setLevel(logging.ERROR)  # Reduce py4j verbosity

# COMMAND ----------

# DBTITLE 1,Load configuration
# Load application configuration for CE processing
config = Config()
config.read_path("../src/config/config.yml")

# Extract CE-specific configuration values
catalog = config.get_config_value("master_creation.catalog")
schema = config.get_config_value("master_creation.schema")
volume = config.get_config_value("master_creation.volume")
cols = config.get_config_value("master_creation.columns_inference")
partition_columns = config.get_config_value("master_creation.partition_columns")
input_table = config.get_config_value("ingestion.table_ingestion")
output_table = config.get_config_value("master_creation.table_inference")
volume_directory = f"/Volumes/{catalog}/{schema}/{volume}/Campagne"

# COMMAND ----------

# DBTITLE 1,Get campaign dates
# Automatically determine current and past campaign dates
currentcampaign_month_year, pastcampaign_month_year = get_campaign_dates(volume_directory)
print("Current campaign:", currentcampaign_month_year)
print("Past campaign:", pastcampaign_month_year)

# COMMAND ----------

# pastcampaign_month_year = '052025'
# currentcampaign_month_year ='062025'

# COMMAND ----------

input_path = f"{catalog}.{schema}.{input_table}"  # _{pastcampaign_month_year}_{currentcampaign_month_year}"
output_path = f"{catalog}.{schema}.{output_table}" #_{pastcampaign_month_year}_{currentcampaign_month_year}"

# COMMAND ----------

# DBTITLE 1,Load and filter CE documents
# Load master table and filter for CE documents only
MasterTable_path = input_path
df = (
    spark.table(MasterTable_path)
    .where(F.col("current_campaign").contains(currentcampaign_month_year))
    .where(F.col("document_type") == "CE")
)
df = df.dropDuplicates()

# COMMAND ----------

# DBTITLE 1,Define CE chunk markers
# Define chunk markers for single commodity CE documents
chunk_markers = [
    "n° plico",
    "Energia Elettrica",
    "Gas Naturale",
    "Vendita di energia elettrica",
    "Vendita di gas naturale",
    "Sconti e/o Bonus",
    "Prodotti e/o servizi aggiuntivi",
    "Tariffa per l'uso della rete dell'energia elettrica",
    "Tariffa per l'uso della rete del gas naturale",
    "Oneri generali di sistema",
    "Ulteriori caratteristiche del Contratto",
    "Modalità di variazione delle Condizioni Economiche"
    ]

chunk_markers_dual = [
    "n° plico",
    "Energia Elettrica",
    "Vendita di energia elettrica",
    "Sconti e/o Bonus",
    "Prodotti e/o servizi aggiuntivi",
    "Tariffa per l'uso della rete dell'energia elettrica",
    "Oneri generali di sistema",
    "Gas Naturale ",
    "Vendita di gas naturale",
    "Sconti e/o Bonus",
    "Prodotti e/o servizi aggiuntivi",
    "Tariffa per l'uso della rete del gas naturale",
    "Oneri generali di sistema",
    "Ulteriori caratteristiche del Contratto",
    "Modalità di variazione delle Condizioni Economiche"]

df_master = TextExtractor(volume_directory).create_master(df, chunk_markers_lvl1=chunk_markers, chunk_markers_dual_lvl1=chunk_markers_dual)

# Validate chunking process - exit if errors found
df_false = df_master.where(F.col("chunk_check") == False)
if df_false.count() > 0:
    print(f"Chunking errors found: {df_false.count()}")
    dbutils.notebook.exit("Error during Chunking")

# COMMAND ----------

# DBTITLE 1,Extract and clean value tables
# COMMAND ----------
df = df_master.withColumn(
    "tabella_valori", F.lit(None).cast(T.ArrayType(T.StringType()))
)
# Extract and clean text, removing stop words from patterns
patterns = [
    r"CE_[A-Z0-9_]+_\d{8}_\d{8}(?:\s*\d/\d)?",
    r"\d{6}[A-Z]+\d{2}[A-Z]+\d{6}"
]
combined_pattern = "(?i)(" + "|".join([f"(?:{p})" for p in patterns]) + ")"

df = (
    df
    .withColumn(
        "stop_words",
        F.when(
            F.col("extracted_chunk").isNull() | (F.length("extracted_chunk") == 0),
            F.array().cast("array<string>")
        ).otherwise(
            F.regexp_extract_all(F.col("extracted_chunk"), F.lit(combined_pattern), 0)
        )
    )
    .withColumn(
        "extracted_chunk_clean",
        F.when(
            F.col("extracted_chunk").isNull() | (F.length("extracted_chunk") == 0),
            F.lit("no text")
        ).otherwise(

                F.regexp_replace(F.col("extracted_chunk"), F.lit(combined_pattern), " ")
            )
        )
    )
df=df.withColumn("extracted_chunk_clean", remove_numbers('extracted_chunk_clean'))
# Filter rows relevant for serving based on content and chunk title
df = df.withColumn(
    "flg_serving",
    filter_serving_rows_udf(F.col("extracted_chunk_clean"), F.col("chunk_title")),
)
# COMMAND ----------
df = df.withColumn(
    "extracted_chunk_clean",
    F.regexp_replace(F.col("extracted_chunk_clean"), r"(?m)^\s*\n", "")
).withColumn(
    "extracted_chunk_clean",
    F.regexp_replace(F.col("extracted_chunk_clean"), r"(?m)^\s+$", "")
)

# COMMAND ----------

# DBTITLE 1,Calculate token count and save
# Calculate token count for each cleaned chunk
df = df.withColumn("token_count", token_count_udf(F.col("extracted_chunk_clean")))
df = df.withColumn("flg_max_tokens", F.when(F.col("token_count") * 5 > 16384, True).otherwise(False))
df = df.withColumn("DATA_SNAPSHOT", F.current_date())
# COMMAND ---------Save Deltatable
cols_list = list(cols.keys())
schema = build_schema({k: cols[k] for k in cols_list}, ("flg_max_tokens", T.BooleanType()))
df = df.selectExpr(*[f"{col} as {col}" for col in schema.fieldNames()])
df.write.format("delta").mode("overwrite").partitionBy(partition_columns[0], partition_columns[1]).option(
    "replaceWhere", "current_campaign LIKE '%" + currentcampaign_month_year + "%' AND document_type = 'CE'"
).saveAsTable(output_path)
