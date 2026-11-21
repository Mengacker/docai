# Databricks notebook source
# MAGIC %md
# MAGIC # DO NOT MODIFY THIS PART AND LEAVE AS IT IS
# MAGIC Fixed section to make MLops working

# COMMAND ----------

# DBTITLE 1,Set notebook extensions
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install -r ../src/requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Set working directory
import os
import sys
from databricks.sdk.runtime import dbutils

notebook_path = "/Workspace/" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
os.chdir(notebook_path)

# allow to use imports from src
sys.path.append("../")

# COMMAND ----------

# MAGIC %md
# MAGIC # EDITABLE SECTION
# MAGIC Here you can put your code, starting from imports

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
from src.master.master_preprocessing import extract_and_clean_udf, token_count_udf, clean_text, extract_tabella_valori, filter_serving_rows_udf
from src.utils.utils import get_campaign_dates, build_schema
import dspy

# COMMAND ----------

# DBTITLE 1,# Initialize logger and Spark session
logger = get_logger()
spark: SparkSession = SparkSession.builder.getOrCreate()
logger_py4j = logging.getLogger("py4j")
logger_py4j.setLevel(logging.ERROR)

# COMMAND ----------

# DBTITLE 1,# Initialize logger and Spark sessionLoad configuration
# Load application configuration for CGC processing
config = Config()
config.read_path("../src/config/config.yml")

# Extract CGC-specific configuration values
catalog = config.get_config_value("master_creation.catalog")
schema = config.get_config_value("master_creation.schema")
volume = config.get_config_value("master_creation.volume")
input_table = config.get_config_value("ingestion.output_table")
output_table = config.get_config_value("master_creation.CGC.output_table_train")
volume_directory = f"/Volumes/{catalog}/{schema}/{volume}/Campagne"
output_path = f"{catalog}.{schema}.{output_table}"
input_path = f"{catalog}.{schema}.{input_table}"

# COMMAND ----------

# DBTITLE 1,# get campaigns dates
currentcampaign_month_year, pastcampaign_month_year = get_campaign_dates(volume_directory)
print("Current campaign:", currentcampaign_month_year)
print("Past campaign:", pastcampaign_month_year)

# COMMAND ----------

MasterTable_path = input_path
df = spark.table(MasterTable_path).where(F.col("current_campaign").contains( currentcampaign_month_year)).where(F.col("document_type") == "CGC")
df = df.dropDuplicates()

# COMMAND ----------

# DBTITLE 1,Extract text from files path
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
            "3) Attività di customer satisfaction e caring",
            "4) Adempimento degli obblighi di legge, regolamenti, di disposizioni di Autorità legittimate dalla legge, addebito del canone RAI) :",
            "5) Accertamento, esercizio o difesa di un diritto del Titolare o di terzi ed in particolare:",
            "6) Invio di comunicazioni promozionali via e-mail su Prodotti e/o Servizi analoghi a quelli forniti (c.d. “soft spam”) :",
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
    dbutils.notebook.exit('Missing chunks')

# COMMAND ----------
# Apply text cleaning and preprocessing to extracted chunks
pattern = r"(?s)Consumo annuo.*\+[\d.,]+%"
incident= "PAST_DOCUMENT"
df=extract_tabella_valori(df_master, "extracted_chunk", "extracted_chunk_clean", pattern, incident)
df = df.withColumn("extracted_chunk_clean", clean_text("extracted_chunk"))

# Extract and clean text, removing stop words
df = (
    df.withColumn("extract_result", extract_and_clean_udf(F.col("extracted_chunk_clean")))
    .withColumn("stop_words", F.col("extract_result")["matches"])
    .withColumn("extracted_chunk_clean", F.col("extract_result")["cleaned_text"])
    .drop("extract_result")
)
# Filter rows relevant for serving based on content and chunk title
df = df.withColumn('flg_serving', filter_serving_rows_udf(F.col("extracted_chunk_clean"), F.col("chunk_title")))

# COMMAND ----------
df = df.withColumn(
    "extracted_chunk_clean",
    F.regexp_replace(F.col("extracted_chunk_clean"), r"(?m)^\s*\n", "")
)
# COMMAND ----------
df = df.withColumn("token_count", token_count_udf(F.col("extracted_chunk_clean")))
df=df.withColumn('flg_max_tokens', F.when(F.col('token_count')*4 > 16384, True).otherwise(False))
# Add snapshot date for versioning
df = df.withColumn("DATA_SNAPSHOT", F.current_date())
