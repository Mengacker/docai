# Databricks notebook source
# MAGIC %md
# MAGIC # SR Document Master Creation Pipeline
# MAGIC ## This notebook processes SR documents for inference, extracting text and creating master tables with special handling for energy comparison tables.

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
from src.master.master_preprocessing import (
    extract_and_clean_SR_udf,
    token_count_udf,
    filter_serving_rows_udf,
    extract_tabella_valori,
)
from src.utils.utils import get_campaign_dates, build_schema

# COMMAND ----------

# DBTITLE 1,Initialize logger and Spark session
# Initialize logging and Spark for SR processing
logger = get_logger()
logger_py4j = logging.getLogger("py4j")
logger_py4j.setLevel(logging.ERROR)  # Reduce py4j verbosity
spark: SparkSession = SparkSession.builder.getOrCreate()

# COMMAND ----------

# DBTITLE 1,Load configuration
# Load application configuration for SR processing
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

# DBTITLE 1,Load and filter SR documents
# Load master table and filter for SR documents only
MasterTable_path = input_path
df = (
    spark.table(MasterTable_path)
    .where(F.col("current_campaign").contains(currentcampaign_month_year))
    .where(F.col("document_type") == "SR")
)
df = df.dropDuplicates()

# COMMAND ----------

# DBTITLE 1,Define SR chunk markers
# Define specific chunk markers for SR (Scheda di Ripielogo) document structure
chunk_markers = [
    "Scheda di confrontabilità per clienti finali domestici di energia elettrica",
    "STIMA DELLA SPESA ANNUA ESCLUSE LE IMPOSTE (IN EURO)",
    "Fasce Orarie",
    "Mix delle fonti energetiche primarie utilizzate per la produzione di energia elettrica (eventuali mancate quadrature dei valori sono dovute agli arrotondamenti)",
    "Modalità di indicizzazione/variazione",
    "Descrizione dello sconto e/o del bonus",
    "Altri dettagli sull'offerta",
    "Altri oneri/Servizi Accessori",
]
df_master = TextExtractor(volume_directory).create_master(df, chunk_markers)

# Validate chunking process - exit if errors found
df_false = df_master.where(F.col("chunk_check") == False)
if df_false.count() > 0 or df_master.count() != df.count() * len(chunk_markers):
    print(f"Chunking errors found: {df_false.count()}")
    dbutils.notebook.exit("Error during Chunking")

# COMMAND ----------

# DBTITLE 1,Extract and clean value tables
# Define pattern for extracting consumption and cost tables
pattern = r"(Consumo annuo[\s\S]*?6\.000\s+.*?%)"  # Pattern to match annual consumption data with percentage increases
incident = "PAST_DOCUMENT"  # Target document section for comparison

# Extract value tables from SR documents using regex pattern
df = extract_tabella_valori(df_master, "extracted_chunk", "extracted_chunk_clean", pattern, incident)

# Apply SR-specific text cleaning and preprocessing
df = (
    df.withColumn("extract_result", extract_and_clean_SR_udf(F.col("extracted_chunk_clean")))
    .withColumn("stop_words", F.col("extract_result")["matches"])
    .withColumn("extracted_chunk_clean", F.col("extract_result")["cleaned_text"])
    .drop("extract_result")
)
date_removal_pattern = r"(\d{1,2}/\d{1,2}/\d{4})(?=[\s\.,]|$)"
df = df.withColumn("extracted_chunk_clean", F.regexp_replace("extracted_chunk_clean", date_removal_pattern, ""))
# Filter rows relevant for serving based on content and chunk title
df = df.withColumn("flg_serving", filter_serving_rows_udf(F.col("extracted_chunk_clean"), F.col("chunk_title")))

# COMMAND ----------

df = df.withColumn("extracted_chunk_clean", F.regexp_replace(F.col("extracted_chunk_clean"), r"(?m)^\s*\n", ""))

# COMMAND ----------

# DBTITLE 1,Calculate token count and save
# Calculate token count for each cleaned chunk
df = df.withColumn("token_count", token_count_udf(F.col("extracted_chunk_clean")))

# Add snapshot date for versioning
df = df.withColumn("DATA_SNAPSHOT", F.current_date())
# Build schema and save to Delta table
cols_list = list(cols.keys())
schema = build_schema({k: cols[k] for k in cols_list})
df = df.selectExpr(*[f"{col} as {col}" for col in schema.fieldNames()])
df.write.format("delta").mode("overwrite").partitionBy(partition_columns[0], partition_columns[1]).option(
    "replaceWhere", "current_campaign LIKE '%" + currentcampaign_month_year + "%' AND document_type = 'SR'"
).saveAsTable(output_path)
