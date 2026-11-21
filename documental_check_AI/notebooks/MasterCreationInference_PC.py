# Databricks notebook source
# MAGIC %md
# MAGIC # PC Document Master Creation Pipeline
# MAGIC ## This notebook processes PC documents for inference, extracting text and creating master tables with chunked content for document comparison.

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
import os
import re
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.window import Window
from src.utils.logger import get_logger
import logging
from src.master.master_creation import TextExtractor
from src.master.master_preprocessing import  token_count_udf, filter_serving_rows_udf
from src.utils.utils import get_campaign_dates, build_schema

# COMMAND ----------

# DBTITLE 1,Initialize logger and Spark session
# Initialize logging and Spark for PC processing
logger = get_logger()
spark: SparkSession = SparkSession.builder.getOrCreate()
logger_py4j = logging.getLogger("py4j")
logger_py4j.setLevel(logging.ERROR)  # Reduce py4j verbosity

# COMMAND ----------

# DBTITLE 1,Load configuration
# Load application configuration for PC processing
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

# DBTITLE 1,Load and filter PC documents
# Load master table and filter for PC documents only
MasterTable_path = input_path
df = spark.table(MasterTable_path).where(F.col("current_campaign").contains( currentcampaign_month_year)).where(F.col("document_type") == "PLICHI")
df = df.dropDuplicates()

# COMMAND ----------

# DBTITLE 1,Define PC chunk markers
# Define specific chunk markers for CGC document structure
chunk_markers = ["Copia Fornitore",
            "Copia Cliente",
            "CONDIZIONI CONTRATTUALI DELL'ATTIVAZIONE"

]
chunk_markers_2 = [
            "Somministrazione di gas naturale -",
            "Somministrazione di energia elettrica -",
            "Somministrazione di gas naturale e energia elettrica -",
            "Dati del cliente",
            "Mandato per l'autorizzazione di addebito in conto delle disposizioni Sepa Core Direct Debit",
            'Elenco punti di Riconsegna',
            'Elenco punti di prelievo',
            'Elenco punti di Prelievo',
            'Elenco Punti di Prelievo e Recesso',
            "Espressione del consenso al trattamento dei dati personali",
            "Documentazione da allegare alla proposta",
            "Condizioni economiche della somministrazione di energia elettrica",
            "Sottoscrizione del Modulo ed espressione del consenso al trattamento dei dati personali",
            "Elenco Punti di Prelievo per cui si richiede il Cambio Prodotto",
            "Dichiarazione di conferimento dei mandati",
            "Formulazione della proposta di contratto",
            "CONTATTI",
            "CONDIZIONI CONTRATTUALI DELL'ATTIVAZIONE"]
df_master = TextExtractor(volume_directory).create_master(df, chunk_markers, chunk_markers_lvl2= chunk_markers_2)
# COMMAND ----------

# Validate chunking process - exit if errors found
df_false = df_master.where(F.col("chunk_check") == False)
if df_false.count() > 0:
    print(f"Chunking errors found: {df_false.count()}")
    dbutils.notebook.exit('Error during Chunking')
if df_master.count() < df.count()*3:
    df_macrosegments= df_master.select("current_file_name","macrosegmento").distinct()

    # Crea un DF con tutti i chunk attesi (explode su markers)
    df_required = df_macrosegments.withColumn(
        "chunk_title", F.explode(F.array([F.lit(x) for x in chunk_markers]))
    )

    # Coppie viste
    df_seen = df_master.select("current_file_name","macrosegmento", "chunk_title").distinct()

    # Mancanti = required LEFT ANTI seen
    df_missing = df_required.join(df_seen, on=["current_file_name","macrosegmento", "chunk_title"], how="left_anti")
    print(f"Missing chunks: {df_missing.select('current_file_name', 'chunk_title').collect()}")
    #dbutils.notebook.exit('Missing chunks')

# COMMAND ----------
# Apply text cleaning and preprocessing to extracted chunks
df = df_master.withColumn(
    "tabella_valori", F.lit(None).cast(T.ArrayType(T.StringType()))
)
# Extract and clean text, removing stop words
patterns = [r"^Proposta_.*_\d{8}", r"Copia Fornitore", r"Copia Cliente"]
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
df = df.withColumn("token_count", token_count_udf(F.col("extracted_chunk_clean")))
df = df.withColumn("flg_max_tokens", F.when(F.col("token_count") * 5 > 16384, True).otherwise(False))
# Add snapshot date for versioning
df = df.withColumn("DATA_SNAPSHOT", F.current_date())

# Build schema and save to Delta table
cols_list = list(cols.keys())
schema = build_schema({k: cols[k] for k in cols_list})
df = df.selectExpr(*[f"{col} as {col}" for col in schema.fieldNames()])
df.write.format("delta").mode("overwrite").partitionBy(partition_columns[0], partition_columns[1]).option(
    "replaceWhere", "current_campaign LIKE '%" + currentcampaign_month_year + "%' AND document_type = 'PC'"
).saveAsTable(output_path)
