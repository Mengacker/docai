# Databricks notebook source
# MAGIC %md
# MAGIC # SS Document Master Creation Pipeline
# MAGIC ## This notebook processes SS documents for inference, extracting text and creating master tables with special handling for energy comparison tables.

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
import re
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
    token_count_udf,
    remove_numbers_in_blocks,
    filter_serving_rows_udf,
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
    .where(F.col("document_type") == "SS")
)
df = df.dropDuplicates()

# COMMAND ----------

# DBTITLE 1,Define SS chunk markers
# Define specific chunk markers for SS (Scheda Sintetica) document structure
chunk_markers = [
    "Scheda sintetica dell'offerta",
    "Spesa annua stimata in €/anno (escluse imposte e tasse)",
    "Condizioni economiche",
    "Altre informazioni",
    "Modalità e oneri per il recesso",
    "Operatore commerciale",
    "Documenti allegati alla scheda tecnica",
    "ALLEGATO",
]
chunk_markers = [str(x) for x in chunk_markers]
df_master = TextExtractor(volume_directory).create_master(df, chunk_markers)

# Validate chunking process - exit if errors found
df_false = df_master.where(F.col("chunk_check") == False)
if df_false.count() > 0:
    print(f"Chunking errors found: {df_false.count()}")
    dbutils.notebook.exit("Error during Chunking")

# COMMAND ----------

# DBTITLE 1,Extract and clean value tables
# Define pattern for extracting consumption and cost tables
df = df_master.withColumn(
    "tabella_valori", F.lit(None).cast(T.ArrayType(T.StringType()))
)
# Extract and clean text, removing stop words
patterns = [r"(Consumo annu?o[\s\S]*?[56]\.000[^\n]*\d{1,3}(?:\.\d{3})*,\d{2})", r'(?s)Valore indice (PUN\s*\*\s*1,1\s*\(€/kWh\)|PSV\s*\(€/Smc\)).*?Indice\s+\1', r"SCHEDA_SINT_[A-Z0-9]+_[A-Z0-9]+_\d{8}_\d{8}",  r"\d{6}[A-Z0-9]+?\d{6}", r"\d{2}/\d{2}/\d{2}"
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

block_re = re.compile(r"Corrispettivi definiti dal venditore[\s\S]*?Vendita")
df = df.withColumn('extracted_chunk_clean', remove_numbers_in_blocks(F.col('extracted_chunk_clean')))
#df=df.withColumn("extracted_chunk_clean", remove_numbers('extracted_chunk_clean'))

# Filter rows relevant for serving based on content and chunk title
df = df.withColumn(
    "flg_serving",
    filter_serving_rows_udf(F.col("extracted_chunk_clean"), F.col("chunk_title")),
)
# COMMAND ----------

df = df.withColumn(
    "extracted_chunk_clean", F.regexp_replace(F.col("extracted_chunk_clean"), r"(?m)^\s*\n", "")
).withColumn("extracted_chunk_clean", F.regexp_replace(F.col("extracted_chunk_clean"), r"(?m)^\s+$", ""))

# COMMAND ----------

# DBTITLE 1,Calculate token count and save
# Calculate token count for each cleaned chunk
df = df.withColumn("token_count", token_count_udf(F.col("extracted_chunk_clean")))
df = df.withColumn("flg_max_tokens", F.when(F.col("token_count") * 5 > 16384, True).otherwise(False))

# Add snapshot date for versioning
df = df.withColumn("DATA_SNAPSHOT", F.current_date())

# Build schema and save to Delta table
cols_list = list(cols.keys())
schema = build_schema({k: cols[k] for k in cols_list})
df = df.selectExpr(*[f"{col} as {col}" for col in schema.fieldNames()])
df.write.format("delta").mode("overwrite").partitionBy(partition_columns[0], partition_columns[1]).option(
    "replaceWhere", "current_campaign LIKE '%" + currentcampaign_month_year + "%' AND document_type = 'SS'"
).saveAsTable(output_path)
