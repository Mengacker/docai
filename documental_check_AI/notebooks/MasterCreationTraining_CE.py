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
import os
import re
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.window import Window
from src.utils.logger import get_logger
import logging
from src.master.master_creation import TextExtractor
from src.master.master_preprocessing import token_count_udf, filter_serving_rows_udf, extract_tabella_valori, remove_numbers, extract_and_clean_multiple_patterns, extract_and_clean_schema
from src.utils.utils import get_campaign_dates, build_schema

# COMMAND ----------

# DBTITLE 1,# Initialize logger and Spark session
logger = get_logger()
spark: SparkSession = SparkSession.builder.getOrCreate()
logger_py4j = logging.getLogger("py4j")
logger_py4j.setLevel(logging.ERROR)

# COMMAND ----------
# DBTITLE 1,# Initialize logger and Spark sessionLoad configuration
config = Config()
config.read_path("../src/config/config.yml")

# Extract CGC-specific configuration values
catalog = config.get_config_value("master_creation.catalog")
schema = config.get_config_value("master_creation.schema")
volume = config.get_config_value("master_creation.volume")
input_table = config.get_config_value("ingestion.output_table")
output_table = config.get_config_value("master_creation.CE.output_table_train")
cols = config.get_config_value("master_creation.CE.columns")
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
df = spark.table(MasterTable_path).where(F.col("current_campaign").contains(currentcampaign_month_year)).where(F.col("document_type") == "CE")
df = df.dropDuplicates()

# COMMAND ----------

# DBTITLE 1,Extract text from files path
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
# COMMAND ----------
df_false = df_master.where(F.col("chunk_check") == False)
if df_false.count() > 0:
    print(f"Chunking errors found: {df_false.count()}")

# COMMAND ----------
df = df_master.withColumn(
    "tabella_valori", F.lit(None).cast(T.ArrayType(T.StringType()))
)
# Extract and clean text, removing stop words
patterns = [r"CE_[A-Z0-9_]+_\d{8}_\d{8}(?:\s*\d/\d)?"]
extract_and_clean_multiple_patterns_udf = F.udf(
    lambda text: extract_and_clean_multiple_patterns(text, patterns),
    extract_and_clean_schema,
)
df = (
    df.withColumn(
        "extract_result",
        extract_and_clean_multiple_patterns_udf(F.col("extracted_chunk")),
    )
    .withColumn("stop_words", F.col("extract_result")["matches"])
    .withColumn("extracted_chunk_clean", F.col("extract_result")["cleaned_text"])
    .drop("extract_result")
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
df = df.withColumn("token_count", token_count_udf(F.col("extracted_chunk_clean")))
df = df.withColumn("flg_max_tokens", F.when(F.col("token_count") * 5 > 16384, True).otherwise(False))
df = df.withColumn("DATA_SNAPSHOT", F.current_date())
df.filter(F.col("flg_max_tokens") == True).count()
# COMMAND ----------
df_training=df.where(F.col('current_file_name').isin(['CE_LFIXATIVSMA020426_GFIXATISMA020426_SMART.pdf']))
df_training.display()

# COMMAND ----------
df_sorted = df_training.where(F.col("flg_serving")== True).orderBy(['macrosegmento','chunk'])
df=df_sorted.withColumn("num_id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
labels_udf = F.udf(lambda indx: ground_truth[indx-1] , T.ArrayType(T.ArrayType(T.StringType())))

# COMMAND ----------

ground_truth = [
    [
        [
            """Energia Elettrica                  codice offerta XXX""",
            """Energia Elettrica                  codice offerta XXX"""
        ],
        [
            """Le presenti Condizioni Economiche sono valide per  mesi di fornitura. Per la somministrazione di energia elettrica  sono fatturati i corrispettivi relativi alla spesa per la Vendita dell'energia elettrica, quelli relativi alla spesa per la Tariffa per l'uso della rete dell'energia elettrica e infine quelli relativi alla spesa per Oneri generali di sistema, come definite dall'Autorità di Regolazione per Energia Reti Ambiente (di seguito "ARERA") nel Glossario della bolletta dei clienti finali di energia. In più, se presenti, qui di seguito sono dettagliati eventuali Sconti e/o Bonus e i corrispettivi di eventuali Prodotti e/o servizi aggiuntivi.""",
            """Le presenti Condizioni Economiche sono valide per  mesi di fornitura. Per la somministrazione di energia elettrica  sono fatturati i corrispettivi relativi alla spesa per la Vendita dell'energia elettrica, quelli relativi alla spesa per la Tariffa per l'uso della rete dell'energia elettrica e infine quelli relativi alla spesa per Oneri generali di sistema, come definite dall'Autorità di Regolazione per Energia Reti Ambiente (di seguito "ARERA") nel Glossario della bolletta dei clienti finali di energia. In più, se presenti, qui di seguito sono dettagliati eventuali Sconti e/o Bonus e i corrispettivi di eventuali Prodotti e/o servizi aggiuntivi."""
        ],
        [
            """Il  Cliente è inoltre tenuto  a  corrispondere i  diritti  e  i tributi  relativi  al  Contratto di  cui  al  relativo Articolo  delle  Condizioni  Generali di Contratto e le cui aliquote sono consultabili sul portale""",
            """Il  Cliente è inoltre tenuto  a  corrispondere i  diritti  e  i tributi  relativi  al  Contratto di  cui  al  relativo Articolo  delle  Condizioni  Generali di Contratto e le cui aliquote sono consultabili sul portale"""
        ]
    ],
    [
        [
            """ Oneri generali di sistema""",
            """ Oneri generali di sistema"""
        ],
        [
            """Si  compongono  dei  corrispettivi  a  carico  del fornitore    a  copertura  di  tali  oneri  definiti  e  aggiornati  da  ARERA,  ivi  compresa  la  componente  Asos  (la componente Asos serve per finanziare il sistema di incentivi riconosciuti per la produzione di energia elettrica da fonti rinnovabili e da cogenerazione. È a carico di tutti i clienti elettrici).""",
            """Si  compongono  dei  corrispettivi  a  carico  del fornitore    a  copertura  di  tali  oneri  definiti  e  aggiornati  da  ARERA,  ivi  compresa  la  componente  Asos  (la componente Asos serve per finanziare il sistema di incentivi riconosciuti per la produzione di energia elettrica da fonti rinnovabili e da cogenerazione. È a carico di tutti i clienti elettrici)."""
        ],
        [
            """Tutti i valori definiti e aggiornati da ARERA sono pubblicati alla pagina a  rera.it/consumatori/valori-rete-oneri-domestici-ee .""",
            """Tutti i valori definiti e aggiornati da ARERA sono pubblicati alla pagina a  rera.it/consumatori/valori-rete-oneri-domestici-ee ."""
        ]
    ],
    [
        [
            """Prodotti e/o servizi aggiuntivi """,
            """Prodotti e/o servizi aggiuntivi """
        ],
        [
            "Non sono previsti prodotti e/o servizi aggiuntivi.",
            "Non sono previsti prodotti e/o servizi aggiuntivi."
        ]
    ],
    [
        [
            """ Sconti e/o Bonus""",
            """ Sconti e/o Bonus """
        ],
        [
            """Il fornitore riconosce lo Sconto Domiciliazione, pari al % del Corrispettivo Luce, applicato nel corso della validità delle presenti condizioni economiche a condizione che l'addebito diretto in conto corrente SEPA risulti attivo. Qualora non sia attivo, il Cliente potrà attivarlo in ogni momento e usufruire dello Sconto Domiciliazione a partire dalla data di attivazione della domiciliazione stessa. Nel caso in cui l'addebito  diretto in conto corrente SEPA venga disattivato per qualunque ragione, lo Sconto Domiciliazione non sarà applicato a partire dalla data di disattivazione della domiciliazione.""",
            """Il fornitore riconosce lo Sconto Domiciliazione, pari al % del Corrispettivo Luce, applicato nel corso della validità delle presenti condizioni economiche a condizione che l'addebito diretto in conto corrente SEPA risulti attivo. Qualora non sia attivo, il Cliente potrà attivarlo in ogni momento e usufruire dello Sconto Domiciliazione a partire dalla data di attivazione della domiciliazione stessa. Nel caso in cui l'addebito  diretto in conto corrente SEPA venga disattivato per qualunque ragione, lo Sconto Domiciliazione non sarà applicato a partire dalla data di disattivazione della domiciliazione."""
        ],
        [
            """L'incidenza percentuale  media  dello  Sconto  Domiciliazione  sulla  stima  della  spesa  annua  ante  imposte  di  un  cliente  tipo  che  consuma  .  kWh  all'anno nell'abitazione di residenza con  kW di potenza impegnata (  ° trimestre  ) è pari a %.""",
            """L'incidenza percentuale  media  dello  Sconto  Domiciliazione  sulla  stima  della  spesa  annua  ante  imposte  di  un  cliente  tipo  che  consuma  .  kWh  all'anno nell'abitazione di residenza con  kW di potenza impegnata (  ° trimestre  ) è pari a %."""
        ],
        [
            """Sconto  Commercializzazione:   il fornitore  riconosce  lo  Sconto  Commercializzazione  pari  a   euro/anno  per  l'intera  durata  delle  Condizioni Economiche;  tale  sconto  è calcolato  sulla  Commercializzazione  e Vendita  dell'offerta  Fixa  Time  listino  Base  pari  a   euro/anno.""",
            """Sconto  Commercializzazione:   il fornitore  riconosce  lo  Sconto  Commercializzazione  pari  a   euro/anno  per  l'intera  durata  delle  Condizioni Economiche;  tale  sconto  è calcolato  sulla  Commercializzazione  e Vendita  dell'offerta  Fixa  Time  listino  Base  pari  a   euro/anno."""
        ],
        [
            """L'incidenza percentuale media del suddetto sconto sulla stima della spesa annua ante imposte di un cliente tipo che consuma . kWh all'anno nell'abitazione di residenza con  kW di potenza impegnata (  ° trimestre )  è pari a  %. """,
            """L'incidenza percentuale media del suddetto sconto sulla stima della spesa annua ante imposte di un cliente tipo che consuma . kWh all'anno nell'abitazione di residenza con  kW di potenza impegnata (  ° trimestre )  è pari a  %. """
        ]
    ],
    [
        [
            """Gas Naturale                      codice offerta XXX""",
            """Gas Naturale                      codice offerta XXX"""
        ],
        [
            """Le presenti Condizioni Economiche sono valide per  mesi di fornitura. Per la somministrazione di gas naturale  sono fatturati i corrispettivi relativi alla spesa per la Vendita di gas naturale, quelli relativi alla spesa per la Tariffa per l'uso della rete del gas naturale e infine quelli relativi alla spesa per Oneri  generali di  sistema, come definite  dall'Autorità di Regolazione per Energia Reti  Ambiente (di seguito  "ARERA") nel  Glossario  della bolletta  dei clienti finali di energia. In più, se presenti, qui di seguito sono dettagliati eventuali Sconti e/o Bonus e i corrispettivi di eventuali Prodotti e/o servizi aggiuntivi. """,
            """Le presenti Condizioni Economiche sono valide per  mesi di fornitura. Per la somministrazione di gas naturale  sono fatturati i corrispettivi relativi alla spesa per la Vendita di gas naturale, quelli relativi alla spesa per la Tariffa per l'uso della rete del gas naturale e infine quelli relativi alla spesa per Oneri  generali di  sistema, come definite  dall'Autorità di Regolazione per Energia Reti  Ambiente (di seguito  "ARERA") nel  Glossario  della bolletta  dei clienti finali di energia. In più, se presenti, qui di seguito sono dettagliati eventuali Sconti e/o Bonus e i corrispettivi di eventuali Prodotti e/o servizi aggiuntivi. """
        ],
        [
            """Il  Cliente è inoltre tenuto  a  corrispondere i  diritti  e  i tributi  relativi  al  Contratto di  cui  al  relativo Articolo  delle  Condizioni  Generali di Contratto e le cui aliquote sono consultabili sul portale.""",
            """Il  Cliente è inoltre tenuto  a  corrispondere i  diritti  e  i tributi  relativi  al  Contratto di  cui  al  relativo Articolo  delle  Condizioni  Generali di Contratto e le cui aliquote sono consultabili sul portale."""
        ]
    ],
    [
        [
            "Oneri generali di sistema",
            "Oneri generali di sistema"
         ],
        [
            """Si compongono dei corrispettivi a carico del fornitore  a copertura di tali oneri definiti e aggiornati da ARERA.""",
            """Si compongono dei corrispettivi a carico del fornitore  a copertura di tali oneri definiti e aggiornati da ARERA."""
        ],
        [
            """Tutti i valori definiti e aggiornati da ARERA sono pubblicati alla pagina a  rera.it/consumatori/valori-rete-oneri-domestici-gas .""",
            """Tutti i valori definiti e aggiornati da ARERA sono pubblicati alla pagina a  rera.it/consumatori/valori-rete-oneri-domestici-gas ."""
        ],
        [
            """Fixa Time Smart""",
            """Fixa Time Smart"""
        ],

        [
            """Condizioni Economiche""",
            """Condizioni Economiche"""
        ],
        [
            """Condizioni del mercato libero sottoscrivibili dal //  al //""",
            """Condizioni del mercato libero sottoscrivibili dal //  al //"""
        ],
        [
            """n° plico""",
            """n° plico"""
        ]
    ],
    [
        ["Prodotti e/o servizi aggiuntivi", "Prodotti e/o servizi aggiuntivi"],
        ["Non sono previsti prodotti e/o servizi aggiuntivi.", "Non sono previsti prodotti e/o servizi aggiuntivi."]
    ],
    [
        ["Sconti e/o Bonus", "Sconti e/o Bonus"],
        ["Il fornitore riconosce lo Sconto Domiciliazione, pari al % del Corrispettivo Gas, applicato nel corso della validità delle presenti condizioni economiche a condizione che l'addebito diretto in conto corrente SEPA risulti attivo. Qualora non sia attivo, il Cliente potrà attivarlo in ogni momento e usufruire dello Sconto Domiciliazione a partire dalla data di attivazione della domiciliazione stessa. Nel caso in cui l'addebito  diretto in conto corrente SEPA venga disattivato per qualunque ragione, lo Sconto Domiciliazione non sarà applicato a partire dalla data di disattivazione della domiciliazione. ",
        "Il fornitore riconosce lo Sconto Domiciliazione, pari al % del Corrispettivo Gas, applicato nel corso della validità delle presenti condizioni economiche a condizione che l'addebito diretto in conto corrente SEPA risulti attivo. Qualora non sia attivo, il Cliente potrà attivarlo in ogni momento e usufruire dello Sconto Domiciliazione a partire dalla data di attivazione della domiciliazione stessa. Nel caso in cui l'addebito  diretto in conto corrente SEPA venga disattivato per qualunque ragione, lo Sconto Domiciliazione non sarà applicato a partire dalla data di disattivazione della domiciliazione. "
        ],
        ["L'incidenza percentuale  media  dello  Sconto  Domiciliazione  sulla  stima  della  spesa  annua  ante  imposte  di  un  cliente  tipo  che  consuma  .  Smc  all'anno nell'ambito tariffario Centro - Sud Occidentale (° trimestre )  è pari a %.",
         "L'incidenza percentuale  media  dello  Sconto  Domiciliazione  sulla  stima  della  spesa  annua  ante  imposte  di  un  cliente  tipo  che  consuma  .  Smc  all'anno nell'ambito tariffario Centro - Sud Occidentale (° trimestre )  è pari a %."],
        ["Sconto  Commercializzazione:  il fornitore  riconosce  lo  Sconto  Commercializzazione  pari  a   euro/anno  per  l'intera  durata  delle  Condizioni Economiche;  tale  sconto  è calcolato  sulla  Commercializzazione  e Vendita  dell'offerta  Fixa  Time  listino  Base  pari  a   euro/anno.  ",
         "Sconto  Commercializzazione:  il fornitore  riconosce  lo  Sconto  Commercializzazione  pari  a   euro/anno  per  l'intera  durata  delle  Condizioni Economiche;  tale  sconto  è calcolato  sulla  Commercializzazione  e Vendita  dell'offerta  Fixa  Time  listino  Base  pari  a   euro/anno.  "],
         ["L'incidenza percentuale  media del  suddetto  sconto  sulla  stima  della  spesa  annua ante  imposte  di  un  cliente tipo  che  consuma  .  Smc  all'anno nell'ambito tariffario Centro-Sud Occidentale (° trimestre ) è pari a   %.",
          "L'incidenza percentuale  media del  suddetto  sconto  sulla  stima  della  spesa  annua ante  imposte  di  un  cliente tipo  che  consuma  .  Smc  all'anno nell'ambito tariffario Centro-Sud Occidentale (° trimestre ) è pari a   %."]
    ],
    [
        ["Modalità di variazione delle Condizioni Economiche", "Modalità di variazione delle Condizioni Economiche"],
        ["Con  un  anticipo  di  almeno  novanta  giorni  rispetto  alla  scadenza  del  loro  periodo  di  validità,  il fornitore  comunicherà  in  forma  scritta  al  Cliente  le nuove  condizioni  economiche  di  somministrazione,  nonché  il  relativo  periodo  di  validità.  La  comunicazione non potrà essere trasmessa all'interno della fattura o congiuntamente a essa, salvo il caso in cui il rinnovo si configuri come una riduzione dei  corrispettivi  originariamente previsti  in  Contratto.", "Con  un  anticipo  di  almeno  novanta  giorni  rispetto  alla  scadenza  del  loro  periodo  di  validità,  il fornitore  comunicherà  in  forma  scritta  al  Cliente  le nuove  condizioni  economiche  di  somministrazione,  nonché  il  relativo  periodo  di  validità.  La  comunicazione non potrà essere trasmessa all'interno della fattura o congiuntamente a essa, salvo il caso in cui il rinnovo si configuri come una riduzione dei  corrispettivi  originariamente previsti  in  Contratto."],
        ["In assenza  di  tale comunicazione,  le  presenti Condizioni  Economiche  s'intenderanno  prorogate fino  a  nuova comunicazione da parte del fornitore, effettuata sempre con un preavviso minimo di novanta giorni. È fatta salva per il Cliente la facoltà di esercitare il  diritto di recesso di cui alle Condizioni Generali di Contratto.", "In assenza  di  tale comunicazione,  le  presenti Condizioni  Economiche  s'intenderanno  prorogate fino  a  nuova comunicazione da parte del fornitore, effettuata sempre con un preavviso minimo di novanta giorni. È fatta salva per il Cliente la facoltà di esercitare il  diritto di recesso di cui alle Condizioni Generali di Contratto."]
    ],
    [
        ["Tariffa per l'uso della rete del gas naturale", "Tariffa per l'uso della rete del gas naturale"],
        ["Si compone dei corrispettivi di importo pari ai costi e agli oneri sostenuti dal fornitore in relazione ai servizi di trasporto, misura, gestione del contatore e distribuzione del gas naturale, che sono definiti e aggiornati da ARERA.", "Si compone dei corrispettivi di importo pari ai costi e agli oneri sostenuti dal fornitore in relazione ai servizi di trasporto, misura, gestione del contatore e distribuzione del gas naturale, che sono definiti e aggiornati da ARERA."]
    ],
    [
        ["Tariffa per l'uso della rete dell'energia elettrica", "Tariffa per l'uso della rete dell'energia elettrica"],
        ["Si compone dei corrispettivi di importo pari ai costi e agli oneri sostenuti dal fornitore in relazione ai servizi di trasporto, misura, gestione del contatore e distribuzione dell'energia elettrica, che sono definiti e aggiornati da ARERA.", "Si compone dei corrispettivi di importo pari ai costi e agli oneri sostenuti dal fornitore in relazione ai servizi di trasporto, misura, gestione del contatore e distribuzione dell'energia elettrica, che sono definiti e aggiornati da ARERA."],
        ["Fixa Time Smart", "Fixa Time Smart"],
        ["Condizioni Economiche", "Condizioni Economiche"],
        ["Condizioni del mercato libero sottoscrivibili dal //  al //", "Condizioni del mercato libero sottoscrivibili dal //  al //"],
        ["n° plico", "n° plico"],
        ["Energia Elettrica                  codice offerta XXX", "Energia Elettrica                  codice offerta XXX"]
    ],
    [
        ["Ulteriori caratteristiche del Contratto", "Ulteriori caratteristiche del Contratto"],
        ["Saranno fatturati al Cliente ulteriori oneri, corrispettivi e/o componenti tariffarie eventualmente introdotti dalle competenti autorità con riferimento alla fornitura di energia elettrica e/o gas naturale a punti di fornitura domestici, anche (ma non esclusivamente) in sostituzione di altri precedentemente previsti dalla normativa di settore. Ad integrazione dell'Art.  delle CGC, in caso di Attivazione o Voltura con cambio fornitore, oltre a qualsiasi altro onere richiesto dal Distributore Gas o dal Distributore Elettrico, verrà applicato un corrispettivo una tantum pari a  euro IVA esclusa. A parziale deroga e laddove non già previsto dall'articolo . delle CGC, l'importo addebitato per ciascuna diffida sarà pari a  € oltre IVA inclusivo sia delle spese postali sia dei costi per attività di  gestione  e  recupero  del  credito,  fatto  salvo  il  risarcimento  del  danno  ulteriore.  Tale  previsione  resterà  valida  ed  efficace  per  l'intera  durata  delle Contratto.", "Saranno fatturati al Cliente ulteriori oneri, corrispettivi e/o componenti tariffarie eventualmente introdotti dalle competenti autorità con riferimento alla fornitura di energia elettrica e/o gas naturale a punti di fornitura domestici, anche (ma non esclusivamente) in sostituzione di altri precedentemente previsti dalla normativa di settore. Ad integrazione dell'Art.  delle CGC, in caso di Attivazione o Voltura con cambio fornitore, oltre a qualsiasi altro onere richiesto dal Distributore Gas o dal Distributore Elettrico, verrà applicato un corrispettivo una tantum pari a  euro IVA esclusa. A parziale deroga e laddove non già previsto dall'articolo . delle CGC, l'importo addebitato per ciascuna diffida sarà pari a  € oltre IVA inclusivo sia delle spese postali sia dei costi per attività di  gestione  e  recupero  del  credito,  fatto  salvo  il  risarcimento  del  danno  ulteriore.  Tale  previsione  resterà  valida  ed  efficace  per  l'intera  durata  delle Contratto."],
        ["Per l'energia elettrica: Per l'intera durata delle presenti Condizioni Economiche il % dell'energia elettrica fornita dal fornitore sarà certificata, tramite garanzie  d'origine  diprovenienza  europea,  come  immessa  in  rete  e  prodotta  da  impianti  alimentati  al  %  da  fonti  rinnovabili,  ai  sensi  della  Delibera ARG/elt", "Per l'energia elettrica: Per l'intera durata delle presenti Condizioni Economiche il % dell'energia elettrica fornita sarà certificata, tramite garanzie  d'origine  diprovenienza  europea,  come  immessa  in  rete  e  prodotta  da  impianti  alimentati  al  %  da  fonti  rinnovabili,  ai  sensi  della  Delibera ARG/elt"],
        ["/.La suddetta Delibera prevede che: i) ogni contratto di vendita di energia rinnovabile debba essere comprovato da una quantità di garanzie di origine (GO) di cui alla direttiva //CE pari alla quantità di energia elettrica venduta come rinnovabile nell'ambito del medesimo contratto; ii) ciascuna società di vendita, entro il  marzo dell'anno successivo a quello in cui è stata fornita energia elettrica ai clienti finali nell'ambito di contratti di vendita energia rinnovabile, sia tenuta ad approvvigionarsi di una quantità di garanzie di origine pari all'energia elettrica venduta come rinnovabile e riferite al medesimo anno, dandone evidenza al GSE secondo modalità da quest'ultimo definite.", "/.La suddetta Delibera prevede che: i) ogni contratto di vendita di energia rinnovabile debba essere comprovato da una quantità di garanzie di origine (GO) di cui alla direttiva //CE pari alla quantità di energia elettrica venduta come rinnovabile nell'ambito del medesimo contratto; ii) ciascuna società di vendita, entro il  marzo dell'anno successivo a quello in cui è stata fornita energia elettrica ai clienti finali nell'ambito di contratti di vendita energia rinnovabile, sia tenuta ad approvvigionarsi di una quantità di garanzie di origine pari all'energia elettrica venduta come rinnovabile e riferite al medesimo anno, dandone evidenza al GSE secondo modalità da quest'ultimo definite."],
        ["Per il gas naturale: Nel caso in cui, ai sensi del D.Lgs.  giugno , n. , come modificato dal D.Lgs.  settembre , n.  in attuazione della direttiva (UE) /, e successive eventuali modificazioni, dovessero entrare in vigore, a partire dal ° gennaio  o successivamente, le previsioni relative agli obblighi  di  acquisto  da  parte  degli  operatori  delle  quote  di  emissioni  di  gas  a  effetto  serra  nell'Unione  Europea,  il fornitore,  fatta  salva  l'applicazione automatica di eventuali interventi tariffari ARERA, avrà il diritto di modificare le presenti Condizioni Economiche includendo gli ulteriori costi dalla stessa sostenuti in relazione all'acquisto di tali quote di emissione. A tali fini, il fornitore invierà, con congruo preavviso, una comunicazione con cui il Cliente sarà debitamente informato dell'entità di tali costi, della ragione della loro applicazione e del momento a partire dal quale verranno addebitati, ferma restando la facoltà per il Cliente di esercitare il diritto di recesso dal Contratto.", "Per il gas naturale: Nel caso in cui, ai sensi del D.Lgs.  giugno , n. , come modificato dal D.Lgs.  settembre , n.  in attuazione della direttiva (UE) /, e successive eventuali modificazioni, dovessero entrare in vigore, a partire dal ° gennaio  o successivamente, le previsioni relative agli obblighi  di  acquisto  da  parte  degli  operatori  delle  quote  di  emissioni  di  gas  a  effetto  serra  nell'Unione  Europea,  il fornitore,  fatta  salva  l'applicazione automatica di eventuali interventi tariffari ARERA, avrà il diritto di modificare le presenti Condizioni Economiche includendo gli ulteriori costi dalla stessa sostenuti in relazione all'acquisto di tali quote di emissione. A tali fini, il fornitore invierà, con congruo preavviso, una comunicazione con cui il Cliente sarà debitamente informato dell'entità di tali costi, della ragione della loro applicazione e del momento a partire dal quale verranno addebitati, ferma restando la facoltà per il Cliente di esercitare il diritto di recesso dal Contratto."]
    ],
    [
        ["Vendita di energia elettrica", "Vendita di energia elettrica"],
        ["Corrispettivi definiti dal venditore", "Corrispettivi definiti dal venditore"],
        ["Corrispettivo a consumo: Corrispettivo Luce , euro/kWh", "Corrispettivo a consumo: Corrispettivo Luce , euro/kWh"],
        ["Se Opzione Bioraria:", "Se Opzione Bioraria:"],
        ["F , euro/kWh", "F , euro/kWh"],
        ["F , euro/kWh", "F , euro/kWh"],
        ["Corrispettivo annuo: Commercializzazione e", "Corrispettivo annuo: Commercializzazione e"],
        ["e  uro/anno", "e  uro/anno"],
        ["Vendita", "Vendita"],
        ["Il Corrispettivo a consumo corrisponde in bolletta al Corrispettivo Luce.", "Il Corrispettivo a consumo corrisponde in bolletta al Corrispettivo Luce."],
        ["Il Corrispettivo Luce inserito in tabella include le Perdite di Rete. In bolletta verrà applicato il valore del Corrispettivo Luce al netto delle Perdite di Rete pari  a ,  €/kWh.  Tale  corrispettivo  sarà  applicato  sia  ai  volumi  di  energia  elettrica  consumata  sia  alla  percentuale  di  Perdite  di  Rete,  definita  e aggiornata da ARERA.", "Il Corrispettivo Luce inserito in tabella include le Perdite di Rete. In bolletta verrà applicato il valore del Corrispettivo Luce al netto delle Perdite di Rete pari  a ,  €/kWh.  Tale  corrispettivo  sarà  applicato  sia  ai  volumi  di  energia  elettrica  consumata  sia  alla  percentuale  di  Perdite  di  Rete,  definita  e aggiornata da ARERA."],
        ["Il Corrispettivo Luce al netto delle Perdite di Rete è invariabile per tutta la durata delle Condizioni Economiche.", "Il Corrispettivo Luce al netto delle Perdite di Rete è invariabile per tutta la durata delle Condizioni Economiche."],
        ["In caso di scelta dell'Opzione Bioraria (codice offerta:  XXX)  sono applicati in fattura due Corrispettivi Luce  distinti  per le  fasce orarie  definite da ARERA: F (dalle : alle :  dal  lunedì al  venerdì escluse festività nazionali) e F (tutte le ore dell'anno non  comprese in fascia F).  In bolletta verranno applicati i valori biorari del Corrispettivo Luce al netto delle Perdite di Rete pari a F , €/kWh e F , €/kWh. Tali corrispettivi saranno applicati sia ai volumi di energia elettrica consumata sia alla percentuale di Perdite di Rete, definita e aggiornata da ARERA.", "In caso di scelta dell'Opzione Bioraria (codice offerta:  XXX)  sono applicati in fattura due Corrispettivi Luce  distinti  per le  fasce orarie  definite da ARERA: F (dalle : alle :  dal  lunedì al  venerdì escluse festività nazionali) e F (tutte le ore dell'anno non  comprese in fascia F).  In bolletta verranno applicati i valori biorari del Corrispettivo Luce al netto delle Perdite di Rete pari a F , €/kWh e F , €/kWh. Tali corrispettivi saranno applicati sia ai volumi di energia elettrica consumata sia alla percentuale di Perdite di Rete, definita e aggiornata da ARERA."],
        ["L'Opzione Bioraria è applicata solo in presenza di misuratore atto a rilevare l'energia per  ciascuna  delle fasce  orarie  F  e F  e in  ogni  caso  a partire dal  momento  in  cui  il  soggetto  responsabile del  servizio  di  misura  rende  effettivamente  disponibili al fornitore i dati di consumo del Cliente aggregati per le due suddette fasce.", "L'Opzione Bioraria è applicata solo in presenza di misuratore atto a rilevare l'energia per  ciascuna  delle fasce  orarie  F  e F  e in  ogni  caso  a partire dal  momento  in  cui  il  soggetto  responsabile del  servizio  di  misura  rende  effettivamente  disponibili al fornitore i dati di consumo del Cliente aggregati per le due suddette fasce."],
        ["Il  Corrispettivo  annuo  è fatturato  mensilmente  in  bolletta  alla  voce  Commercializzazione  e Vendita,  invariabile  per  tutta  la  durata  delle  Condizioni Economiche.", "Il  Corrispettivo  annuo  è fatturato  mensilmente  in  bolletta  alla  voce  Commercializzazione  e Vendita,  invariabile  per  tutta  la  durata  delle  Condizioni Economiche."],
        ["Corrispettivo definito da ARERA:  si  applica, inoltre, il corrispettivo di dispacciamento pari  al valore del corrispettivo  CdispD , definito da ARERA nel Testo integrato delle disposizioni per l'erogazione dei servizi di vendita dell'energia elettrica di ultima istanza ed espresso in €/kWh.", "Corrispettivo definito da ARERA:  si  applica, inoltre, il corrispettivo di dispacciamento pari  al valore del corrispettivo  CdispD , definito da ARERA nel Testo integrato delle disposizioni per l'erogazione dei servizi di vendita dell'energia elettrica di ultima istanza ed espresso in €/kWh."]
    ],
    [
        ["Vendita di gas naturale", "Vendita di gas naturale"],
        ["Corrispettivi definiti dal venditore", "Corrispettivi definiti dal venditore"],
        ["Corrispettivo a consumo: Corrispettivo Gas ,  euro/Smc Corrispettivo annuo: Commercializzazione e", "Corrispettivo a consumo: Corrispettivo Gas ,  euro/Smc Corrispettivo annuo: Commercializzazione e"],
        ["euro/anno", "euro/anno"],
        ["Vendita", "Vendita"],
        ["Il Corrispettivo a consumo corrisponde in bolletta al Corrispettivo Gas.", "Il Corrispettivo a consumo corrisponde in bolletta al Corrispettivo Gas."],
        ["Il Corrispettivo Gas è invariabile per tutta la durata delle Condizioni Economiche, esso è applicato al gas naturale prelevato con riferimento ad un potere calorifico superiore convenzionale \"P\" (PCS) pari a , GJ/Smc.", "Il Corrispettivo Gas è invariabile per tutta la durata delle Condizioni Economiche, esso è applicato al gas naturale prelevato con riferimento ad un potere calorifico superiore convenzionale \"P\" (PCS) pari a , GJ/Smc."],
        ["In sede di fatturazione, il Corrispettivo Gas sarà applicato ai volumi  con  riferimento  al  potere  calorifico  superiore  convenzionale “P” (PCS)  relativo all'impianto  di  distribuzione  cui  è  connesso  il  Punto di Fornitura,  secondo  le  disposizioni del Testo integrato delle attività di vendita al dettaglio di gas naturale e gas diversi da gas naturale (di seguito, “TIVG”).  Nel caso in cui il Punto di Fornitura non sia dotato di apparecchiatura per la correzione delle misure alle condizioni standard, la correzione ai fini della fatturazione dei volumi misurati avverrà in base al valore del coefficiente C applicabile, secondo le disposizioni del TIVG.", "In sede di fatturazione, il Corrispettivo Gas sarà applicato ai volumi  con  riferimento  al  potere  calorifico  superiore  convenzionale “P” (PCS)  relativo all'impianto  di  distribuzione  cui  è  connesso  il  Punto di Fornitura,  secondo  le  disposizioni del Testo integrato delle attività di vendita al dettaglio di gas naturale e gas diversi da gas naturale (di seguito, “TIVG”).  Nel caso in cui il Punto di Fornitura non sia dotato di apparecchiatura per la correzione delle misure alle condizioni standard, la correzione ai fini della fatturazione dei volumi misurati avverrà in base al valore del coefficiente C applicabile, secondo le disposizioni del TIVG."],
        ["Il Corrispettivo annuo è fatturato mensilmente in bolletta alla voce Commercializzazione e Vendita, invariabile per tutta la durata delle Condizioni Economiche.", "Il Corrispettivo annuo è fatturato mensilmente in bolletta alla voce Commercializzazione e Vendita, invariabile per tutta la durata delle Condizioni Economiche."]
    ],
    [
        ["n° plico", "n° plico"],
        ["In caso di somministrazione di energia elettrica, il Cliente dichiara sotto la propria responsabilità che il PdP oggetto del Contratto rifornisce un'abitazione e eventuali locali annessi collegati allo stesso PdP con potenza impegnata non superiore a  kW.", "In caso di somministrazione di energia elettrica, il Cliente dichiara sotto la propria responsabilità che il PdP oggetto del Contratto rifornisce un'abitazione e eventuali locali annessi collegati allo stesso PdP con potenza impegnata non superiore a  kW."]
    ]
]
# COMMAND -------------
df_sorted = df_training.where(F.col("flg_serving") == True).orderBy("chunk")
df = df_sorted.withColumn(
    "num_id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
)
#COMMAND -------------
labels_udf = F.udf(
    lambda indx: ground_truth[indx - 1], T.ArrayType(T.ArrayType(T.StringType()))
)
df_target = df.withColumn("ground_truth", labels_udf("num_id"))
# COMMAND -------------
extra_column=('ground_truth',
T.ArrayType(
            T.ArrayType(T.StringType())))
cols_list = list(cols.keys())[1:15]
schema = build_schema({k: cols[k] for k in cols_list}, extra_col=extra_column)
df = df_target.selectExpr(*[f"{col} as {col}" for col in schema.fieldNames()])
# COMMAND -------------

df.write.option("overwriteSchema", "true").mode("overwrite").format("delta").saveAsTable(output_path)
