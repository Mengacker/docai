# Databricks notebook source
# MAGIC %md
# MAGIC # CGC Document Master Creation Pipeline
# MAGIC ## This notebook processes CGC documents for inference, extracting text and creating master tables with chunked content for document comparison.

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
from src.utils.logger import get_logger
import logging
from src.master.master_creation import TextExtractor
from src.master.master_preprocessing import  token_count_udf, clean_text, extract_tabella_valori, filter_serving_rows_udf
from src.utils.utils import get_campaign_dates, build_schema

# COMMAND ----------

# DBTITLE 1,Initialize logger and Spark session
# Initialize logging and Spark for CGC processing
logger = get_logger()
spark: SparkSession = SparkSession.builder.getOrCreate()
logger_py4j = logging.getLogger("py4j")
logger_py4j.setLevel(logging.ERROR)  # Reduce py4j verbosity

# COMMAND ----------

# DBTITLE 1,Load configuration
# Load application configuration for CGC processing
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

input_path = f"{catalog}.{schema}.{input_table}"  # _{pastcampaign_month_year}_{currentcampaign_month_year}"
output_path = f"{catalog}.{schema}.{output_table}" #_{pastcampaign_month_year}_{currentcampaign_month_year}"
# COMMAND ----------

# DBTITLE 1,Load and filter CGC documents
# Load master table and filter for CGC documents only
MasterTable_path = input_path
df = spark.table(MasterTable_path).where(F.col("current_campaign").contains( currentcampaign_month_year)).where(F.col("document_type") == "CGC")
df = df.dropDuplicates()

# COMMAND ----------

# DBTITLE 1,Define CGC chunk markers
# Define specific chunk markers for CGC document structure
# Define specific chunk markers for CGC document structure
chunk_markers = [
            "Livelli di qualità commerciale dei servizi di vendita di energia elettrica e gas ai sensi del TIQV",
            "Art. 1",
            "Modulo ripensamento:",
            "Art. 2",
            "Art. 3",
            "Accettazione - L’accettazione della Proposta di Contratto da parte del fornitore è subordinata ad attività di verifica dell’affidabilità creditizia basate:" ,
            "Inizio delle somministrazioni; condizione risolutiva;",
            "Se il Cliente ha richiesto l’esecuzione anticipata del Contratto,",
            "Art. 4",
            "Art. 5",
            "Art. 6",
            "Art. 7",
            "Art. 8",
            "8.3. Azioni in caso di mancato o parziale pagamento",
            "Per i PdF connessi in bassa tensione,",
            "Art. 9",
            "Art. 10",
            "Art. 11",
            "Art. 12",
            "Art. 13",
            "Art. 14",
            "Art. 15",
            "Art. 16",
            "Art. 17",
            "Art. 18",
            "Art. 19",
            "Art. 20",
            "Art. 21",
            "Art. 22",
            "Modulo per l'esercizio del ripensamento",
            "Informativa sul trattamento dati personali dei Clienti",
            "1) Esecuzione di misure precontrattuali su richiesta del Cliente, relative a:",
            "2) Esecuzione delle obbligazioni contrattuali ed in particolare:",
            "3) Attività di customer satisfaction",
            "4) Adempimento degli obblighi di legge, regolamenti, di disposizioni di Autorità legittimate dalla legge, addebito del canone RAI) :",
            "5) Accertamento, esercizio o difesa di un diritto del Titolare o di terzi ed in particolare:",
            "6) Invio di comunicazioni promozionali via e-mail su Prodotti e/o Servizi del fornitore analoghi a quelli forniti (c.d. “soft spam”) :",
            "7) Attività di marketing relative a future iniziative curate dal fornitore su Prodotti e/o Servizi propri e di partner commerciali via SMS, e-mail o telefono con operatore:",
            "8) Attività di marketing, previa comunicazione dei dati da parte della Società, relative a future iniziative promozionali curate da altre imprese su propri Prodotti e/o Servizi via SMS, e-mail o telefono con operatore:",
            "9) Analisi e ricerche di mercato effettuate direttamente dal fornitore",
            "Informazioni Experian e descrizioni"
        ]
df_master = TextExtractor(volume_directory).create_master(df, chunk_markers)
# COMMAND ----------

# Validate chunking process - exit if errors found
df_false = df_master.where(F.col("chunk_check") == False)
if df_false.count() > 0:
    print(f"Chunking errors found: {df_false.count()}")
    dbutils.notebook.exit('Error during Chunking')
if df_master.count() != df.count()*len(chunk_markers):
    df_macrosegments= df_master.select("macrosegmento").distinct()

    # Crea un DF con tutti i chunk attesi (explode su markers)
    df_required = df_macrosegments.withColumn(
        "chunk_title", F.explode(F.array([F.lit(x) for x in chunk_markers]))
    )

    # Coppie viste
    df_seen = df_master.select("macrosegmento", "chunk_title").distinct()

    # Mancanti = required LEFT ANTI seen
    df_missing = df_required.join(df_seen, on=["macrosegmento", "chunk_title"], how="left_anti")
    print(f"Missing chunks: {df_missing.select('chunk_title').collect()}")
    #dbutils.notebook.exit('Missing chunks')

# COMMAND ----------
# Apply text cleaning and preprocessing to extracted chunks
pattern = r"(?s)Consumo annuo.*\+[\d.,]+%"
incident= "PAST_DOCUMENT"
df=extract_tabella_valori(df_master, "extracted_chunk", "extracted_chunk_clean", pattern, incident)
df = df.withColumn("extracted_chunk_clean", clean_text("extracted_chunk"))

# Extract and clean text, removing stop words
patterns = [r"CGC_[A-Z]+_\d{8}\.doc"]

combined_pattern = "(?i)(" + "|".join([f"(?:{p})" for p in patterns]) + ")"

df = (
    df
    .withColumn(
        "stop_words",
        F.when(
            F.col("extracted_chunk_clean").isNull() | (F.length("extracted_chunk_clean") == 0),
            F.array().cast("array<string>")
        ).otherwise(
            F.regexp_extract_all(F.col("extracted_chunk_clean"), F.lit(combined_pattern), 0)
        )
    )
    .withColumn(
        "extracted_chunk_clean",
        F.when(
            F.col("extracted_chunk_clean").isNull() | (F.length("extracted_chunk_clean") == 0),
            F.lit("no text")
        ).otherwise(

                F.regexp_replace(F.col("extracted_chunk_clean"), F.lit(combined_pattern), " ")
            )
        )
    )
# Filter rows relevant for serving based on content and chunk title
df = df.withColumn('flg_serving', filter_serving_rows_udf(F.col("extracted_chunk_clean"), F.col("chunk_title")))

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
    "replaceWhere", "current_campaign LIKE '%" + currentcampaign_month_year + "%' AND document_type = 'CGC'"
).saveAsTable(output_path)
