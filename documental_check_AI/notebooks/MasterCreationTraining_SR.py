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
# MAGIC %pip install -r /Workspace/.../src/requirements.txt
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
from pyspark.sql.window import Window
import pyspark.sql.types as T
import pyspark.sql.functions as F
from src.utils.logger import get_logger
from src.master.master_creation import TextExtractor
from src.master.master_preprocessing import (
    extract_and_clean_SR_udf,
    token_count_udf,
    clean_text,
    remove_numbers,
    filter_serving_rows_udf,
    extract_tabella_valori
)
from src.utils.utils import get_campaign_dates, build_schema


# COMMAND ----------

# DBTITLE 1,# Initialize logger and Spark session
logger = get_logger()
spark: SparkSession = SparkSession.builder.getOrCreate()
logger_py4j = logging.getLogger("py4j")
logger_py4j.setLevel(logging.ERROR)

# COMMAND ----------
# DBTITLE 1,Load configuration
# Load application configuration for SR processing
config = Config()
config.read_path("../src/config/config.yml")

# Extract SR-specific configuration values
catalog = config.get_config_value("master_creation.catalog")
schema = config.get_config_value("master_creation.schema")
volume = config.get_config_value("master_creation.volume")
cols = config.get_config_value("master_creation.columns_inference")
input_table = config.get_config_value("ingestion.output_table")
output_table = config.get_config_value("master_creation.SR.output_table_train")

# Build directory and table paths
volume_directory = f"/Volumes/{catalog}/{schema}/{volume}"
# COMMAND ----------
# DBTITLE 1,Get campaign dates
# Automatically determine current and past campaign dates
#currentcampaign_month_year, pastcampaign_month_year = get_campaign_dates(volume_directory)
#print("Current campaign:", currentcampaign_month_year)
#print("Past campaign:", pastcampaign_month_year)
pastcampaign_month_year = '072025'
currentcampaign_month_year ='082025'
# COMMAND ----------
# DBTITLE 1,Build table paths
# Construct input and output table paths with campaign dates
input_path = f"{catalog}.{schema}.{input_table}"#_{pastcampaign_month_year}_{currentcampaign_month_year}"
output_path = f"{catalog}.{schema}.{output_table}_{pastcampaign_month_year}_{currentcampaign_month_year}"

# COMMAND ----------
MasterTable_path= f'{input_path}'
df = spark.table(MasterTable_path).where(F.col("current_campaign").contains(currentcampaign_month_year)).where(F.col("document_type") == "SR")
df=df.where(F.col('current_file_name').isin(['SR_LUCE_LPLFIXRBAS110725_BASE.pdf']))
df=df.dropDuplicates()


# COMMAND ------------
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
if df_false.count() > 0:
    print(f"Chunking errors found: {df_false.count()}")
    dbutils.notebook.exit('Error during Chunking')
# COMMAND -------------

# DBTITLE 1,Extract and clean value tables
# Define pattern for extracting consumption and cost tables
pattern = r"(?s)Consumo annuo.*\+[\d.,]+%"  # Pattern to match annual consumption data with percentage increases
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
date_removal_pattern = r'(\d{1,2}/\d{1,2}/\d{4})(?=[\s\.,]|$)'
df = df.withColumn("extracted_chunk_clean", F.regexp_replace("extracted_chunk_clean", date_removal_pattern, ""))
# Filter rows relevant for serving based on content and chunk title
df = df.withColumn('flg_serving', filter_serving_rows_udf(F.col("extracted_chunk_clean"), F.col("chunk_title")))
# COMMAND -------------
# DBTITLE 1,Calculate token count and save
# Calculate token count for each cleaned chunk
df = df.withColumn("token_count", token_count_udf(F.col("extracted_chunk_clean")))

# Add snapshot date for versioning
df = df.withColumn("DATA_SNAPSHOT", F.current_date())
df=df.filter(df.flg_serving == True)

# Build schema and save to Delta table
#cols_list = list(cols.keys())
#schema = build_schema({k: cols[k] for k in cols_list})
#df = df.selectExpr(*[f"{col} as {col}" for col in schema.fieldNames()])
# COMMAND -------------
# DBTITLE 1,Add ground truth labels
# Add ground truth labels for training
ground_truth=[

[
    [
      "Scheda   di confrontabilità  per clienti finali domestici di energia  elettrica",
      "Scheda   di confrontabilità  per clienti finali domestici di energia  elettrica"
    ],
    [
      "Corrispettivi previsti dall'offerta PLACET FISSA LUCE DOMESTICO del forniture alla data del , valida fino alla data del .",
      "Corrispettivi previsti dall'offerta PLACET FISSA LUCE DOMESTICO del fornitore alla data del , valida fino alla data del ."
    ],
    [
      "Le condizioni economiche e contrattuali del servizio di maggior tutela, la cui stima di spesa annua è riportata nella presente scheda, sono disponibili solo per i clienti vulnerabili, come definiti dalla normativa vigente. Per ottenere ulteriori informazioni consulti il sito dell'Autorità di Regolazione per Energia Reti e Ambiente w ww.arera.it/consumatori o chiami il numero verde 800.166.654.",
      "Le condizioni economiche e contrattuali del servizio di maggior tutela, la cui stima di spesa annua è riportata nella presente scheda, sono disponibili solo per i clienti vulnerabili, come definiti dalla normativa vigente. Per ottenere ulteriori informazioni consulti il sito dell'Autorità di Regolazione per Energia Reti e Ambiente w ww.arera.it/consumatori o chiami il numero verde 800.166.654."
    ]
],
[
    [
      "STIMA DELLA SPESA ANNUA ESCLUSE LE IMPOSTE (IN EURO)",
      "STIMA DELLA SPESA ANNUA ESCLUSE LE IMPOSTE (IN EURO)"
    ],
    [
      "I valori indicati in tabella, calcolati alla data dell'offerta, possono subire variazioni, nel periodo di validità dell'offerta, a seguito di provvedimenti di ARERA. Tali valori sono stati determinati considerando la seguente ripartizione percentuale dei consumi nelle diverse fasce orarie: F1 33%, F2 31% e F3 36%.",
      "I valori indicati in tabella, calcolati alla data dell'offerta, possono subire variazioni, nel periodo di validità dell'offerta, a seguito di provvedimenti di ARERA. Tali valori sono stati determinati considerando la seguente ripartizione percentuale dei consumi nelle diverse fasce orarie: F1 33%, F2 31% e F3 36%."
    ],

    [
      "Per una stima della spesa annua personalizzata sulla base del Suo consumo effettivo può utilizzare il \"Portale delle Offerte Luce e Gas\".",
      "Per una stima della spesa annua personalizzata sulla base del Suo consumo effettivo può utilizzare il \"Portale delle Offerte Luce e Gas\"."
    ],
],
[
    [
      "Fasce Orarie",
      "Fasce Orarie"
    ],
    [
      "Fascia F1         Dalle 8 alle 19 dei giorni feriali",
      "Fascia F1         Dalle 8 alle 19 dei giorni feriali"
    ],
    [
      "Fascia F2 e F3    Dalle 19 alle 8 dei giorni feriali e tutte le ore del sabato, della domenica e dei giorni festivi",
      "Fascia F2 e F3    Dalle 19 alle 8 dei giorni feriali e tutte le ore del sabato, della domenica e dei giorni festivi"
    ]
],
[
    [
      "Mix delle fonti energetiche primarie utilizzate per la produzione di energia elettrica (eventuali mancate quadrature dei valori sono dovute agli arrotondamenti)",
      "Mix delle fonti energetiche primarie utilizzate per la produzione di energia elettrica (eventuali mancate quadrature dei valori sono dovute agli arrotondamenti)"
    ],

    [
        "Fonti Primarie utilizzate",
        "Fonti Primarie utilizzate"
    ],
    [
        "Composizione del mix energetico per il presente contratto",
        "Composizione del mix energetico per il presente contratto"
    ],
    [
        "Composizione del mix iniziale nazionale utilizzato per la produzione di energia elettrica immessa nel sistema elettrico italiano (%)*",
        "Composizione del mix iniziale nazionale utilizzato per la produzione di energia elettrica immessa nel sistema elettrico italiano (%)*"
    ],
    [
        "Composizione del mix energetico nazionale utilizzato per la produzione di energia elettrica venduta dal fornitore ai clienti finali (%)",
        "Composizione del mix energetico nazionale utilizzato per la produzione di energia elettrica venduta dal fornitore ai clienti finali (%)"
    ],
    [
        "Fonti rinnovabili 100,00%           51,83%            96,34%",
        "Fonti rinnovabili       90,78%                  46,31%"
    ],
    [
        "Carbone           0,00%             1,52%             0,47%",
        "Carbone                 1,88%                   5,27%"
    ],
    [
        "Gas Naturale      0,00%             42,01%            2,65%",
        "Gas naturale            6,18%                   42,99%"
    ],
    [
        "Prodotti petroliferi 0,00%          0,47%             0,04%",
        "Prodotti petroliferi    0,16%                   0,90%"
    ],
    [
        "Nucleare          0,00%             0,00%             0,20%",
        "Nucleare                0,30%                   0,00%"
    ],
    [
        "Lignite           0,00%             0,00%             0,00%",
        "Lignite                 N.D.                    N.D."
    ],
    [
        "Altre fonti       0,00%             4,17%             0,30%",
        "Altre fonti             0,70%                   4,53%"
    ],
    [
        "*la pubblicazione, in base alle previsioni di cui all'articolo 8 del Decreto del Ministro dell'Ambiente e della Sicurezza Energetica del 14 luglio 2023, riporta le informazioni pubblicate dal Gestore dei Servizi Energetici (GSE) sul proprio sito internet e sul portale Fuel Mix.",
        ""
    ],
    [
        "Per dettagli sulle modalità di calcolo si rimanda alla procedura 'PROCEDURA PER LA DETERMINAZIONE DEL MIX ENERGETICO UTILIZZATO PER LA PRODUZIONE DELL'ENERGIA ELETTRICA VENDUTA DALL'IMPRESA DI VENDITA' pubblicata sul sito del Gestore dei Servizi Energetici (GSE).",
        "*Per dettagli sulle modalità di calcolo si rimanda alla procedura “PROCEDURA PER LA DETERMINAZIONE DEL MIX ENERGETICO UTILIZZATO PER LA PRODUZIONE DELL'ENERGIA ELETTRICA VENDUTA DALL'IMPRESA DI VENDITA” pubblicata sul sito del Gestore dei Servizi Energetici (GSE)."
    ]
],
[
    [
        "Modalità di indicizzazione/variazione",
        "Modalità di indicizzazione/variazione"
    ],
    [
        "I corrispettivi sono soggetti ad eventuali indicizzazioni/variazioni definite da ARERA, qualora applicabili.",
        "I corrispettivi sono soggetti ad eventuali indicizzazioni/variazioni definite da ARERA, qualora applicabili."
    ],
],
[
    [
        "Altri oneri/Servizi Accessori",
        "Altri oneri/Servizi Accessori"
    ],
    [
        "Costi non opzionali, applicati al verificarsi di casi particolari:",
        "Costi non opzionali, applicati al verificarsi di casi particolari:"
    ],
    [
        "- per Voltura: 23 euro IVA esclusa",
        "- per Voltura: 23 euro IVA esclusa"
    ]
]
]
# COMMAND -------------
# Build conditional ordering expression
order_expr = F.when(F.col("chunk_title") == chunk_markers[0], 0)
for i, val in enumerate(chunk_markers[1:], start=1):
    order_expr = order_expr.when(F.col("chunk_title") == val, i)

# Apply ordering
df_sorted = df.orderBy(order_expr)
df=df_sorted.withColumn("num_id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
labels_udf = F.udf(lambda indx: ground_truth[indx-1] , T.ArrayType(T.ArrayType(T.StringType())))
df_target = df.withColumn('ground_truth',
                       labels_udf('num_id'))
# COMMAND -------------
extra_column=('ground_truth',
T.ArrayType(
            T.ArrayType(T.StringType())))
cols_list = list(cols.keys())
schema = build_schema({k: cols[k] for k in cols_list},extra_col=extra_column)
df = df_target.selectExpr(*[f"{col} as {col}" for col in schema.fieldNames()])
df.write.option("overwriteSchema", "true").mode("overwrite").format("delta").saveAsTable(output_path)
