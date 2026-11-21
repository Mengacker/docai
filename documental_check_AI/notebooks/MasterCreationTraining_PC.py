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
from src.master.master_preprocessing import extract_and_clean_multiple_patterns, token_count_udf, filter_serving_rows_udf
from src.utils.utils import get_campaign_dates, build_schema
from src.master.master_preprocessing import extract_and_clean_schema

# COMMAND ----------

# DBTITLE 1,# Initialize logger and Spark session
logger = get_logger()
spark: SparkSession = SparkSession.builder.getOrCreate()
logger_py4j = logging.getLogger("py4j")
logger_py4j.setLevel(logging.ERROR)

# COMMAND ----------

# DBTITLE 1,# Initialize logger and Spark sessionLoad configuration
# Load application configuration for CGC processing
# Load application configuration for CGC processing
config = Config()
config.read_path("../src/config/config.yml")

# Extract CGC-specific configuration values
catalog = config.get_config_value("master_creation.catalog")
schema = config.get_config_value("master_creation.schema")
volume = config.get_config_value("master_creation.volume")
input_table = config.get_config_value("ingestion.output_table")
output_table = config.get_config_value("master_creation.PC.output_table_train")
cols = config.get_config_value("master_creation.PC.columns")
volume_directory = f"/Volumes/{catalog}/{schema}/{volume}/Campagne"
output_path = f"{catalog}.{schema}.{output_table}"
input_path = f"{catalog}.{schema}.{input_table}"

# COMMAND ----------

# DBTITLE 1,# get campaigns dates
currentcampaign_month_year= '012026'
pastcampaign_month_year= '122025'


# COMMAND ----------
# DBTITLE 1,Load Master Table

MasterTable_path = input_path
df = spark.table(MasterTable_path).where(F.col("current_campaign").contains(currentcampaign_month_year)).where(F.col("document_type") == "PLICHI")
df = df.dropDuplicates()

# COMMAND ----------

# DBTITLE 1,Extract text from files path
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
df=df.withColumn('flg_max_tokens', F.when(F.col('token_count')*5 > 16384, True).otherwise(False))
# Add snapshot date for versioning
df = df.withColumn("DATA_SNAPSHOT", F.current_date())
# COMMAND ----------
# DBTITLE 1, Insert Ground Truth Column
df_training=df.where((F.col("current_file_name").startswith("ATTCART_GAS_BASE")) & (F.col("chunk_title").startswith("Copia Fornitore")) )
ground_truth = [
    [
        [
            """servizio clienti:
      000.000.000 da rete fissa
      e 00.000.000 da rete mobile al costo previsto dall'operatore telefonico
      portale
      Società S.p.A., Casella Postale n.
      cap città (XX)""",
            """servizio clienti:
      000.000.000 da rete fissa
            e 00.000.000 da rete mobile al costo previsto dall'operatore telefonico
            portale
            Società S.p.A., Casella Postale n.
            cap città (XX)"""
        ],
        ["Proposta di contratto per la casa", "Proposta di contratto per la casa"],
        [
            "Codice Convenzione\n Codice incaricato \nN° Plico",
            "Codice Convenzione\n Codice incaricato \nN° Plico"
        ]
    ],
    [
        [
            """Nome e Cognome*                      Residente a
      Codice Fiscale*
      Documento d'identità Carta d'identità Patente   Altro (specificare)
      N. Documento            Rilasciato da (Ente)        Data Rilascio*
      Telefono             Cellulare      Email
      Indirizzo (Via/Corso)                 Numero       Abitazione di Residenza      si      no
      Fornitura  CAP     Comune                                   Provincia
      Indirizzo (Via/Corso)                 Numero
      Invio Fatture    CAP    Comune                            Provincia
      Indirizzo (Via/Corso)                 Numero
      Residenza cliente*   CAP    Comune                         Provincia  """,
            """Nome e Cognome*                      Residente a
      Codice Fiscale*
      Documento d'identità Carta d'identità Patente   Altro (specificare)
      N. Documento            Rilasciato da (Ente)        Data Rilascio*
      Telefono             Cellulare      Email
      Indirizzo (Via/Corso)                 Numero       Abitazione di Residenza      si      no
      Fornitura  CAP     Comune                                   Provincia
      Indirizzo (Via/Corso)                 Numero
      Invio Fatture    CAP    Comune                            Provincia
      Indirizzo (Via/Corso)                 Numero
      Residenza cliente*   CAP    Comune                         Provincia  """
        ],
        [
            """Chiede l'attivazione della somministrazione di Gas
         Chiede l'attivazione della somministrazione di Gas subordinata all'esecuzione dei lavori
      Indicare il Codice Offerta*
      Codice PDR*                     Matricola contatore*
      Uso:   Cottura Riscaldamento Acqua calda Altri usi Consumo annuo (Smc)
      Tipo abitazione*:     In condominio (più di 6 unità abitative) In stabile uni-plurifamiliare (massimo 6 unità abitative)""",
            """Chiede l'attivazione della somministrazione di Gas
         Chiede l'attivazione della somministrazione di Gas subordinata all'esecuzione dei lavori
      Indicare il Codice Offerta*
      Codice PDR*                     Matricola contatore*
      Uso:   Cottura Riscaldamento Acqua calda Altri usi Consumo annuo (Smc)
      Tipo abitazione*:     In condominio (più di 6 unità abitative) In stabile uni-plurifamiliare (massimo 6 unità abitative)"""
        ],
        [
            """ Chiede l'esecuzione anticipata del contratto
      il Cliente, consapevole degli effetti sulla fornitura dell'eventuale esercizio del diritto di ripensamento, indicati in dettaglio all'Art. 3 delle
      CGC,  chiede  al fornitore  di  avviare  le  attività  necessarie  per  l'esecuzione  del  Contratto  prima  della  scadenza  del  termine  di
      ripensamento.""",
            """ Chiede l'esecuzione anticipata del contratto
      il Cliente, consapevole degli effetti sulla fornitura dell'eventuale esercizio del diritto di ripensamento, indicati in dettaglio all'Art. 3 delle
      CGC,  chiede  al fornitore  di  avviare  le  attività  necessarie  per  l'esecuzione  del  Contratto  prima  della  scadenza  del  termine  di
      ripensamento."""
        ],
        [
            """ Comunicazioni e bolletta digitale
      Il  Cliente  accetta:  1)  di  ricevere  la  documentazione  contrattuale  (inclusi  le  informazioni  precontrattuali  e il  Contratto)  e le
      comunicazioni relative al Contratto in formato elettronico all'indirizzo email indicato nel Modulo, fatto salvo successivi aggiornamenti
      comunicati al fornitore da parte del Cliente. Tale consenso del Cliente vale anche per tutte le successive comunicazioni per le quali la
      legge o il Contratto non preveda una diversa e specifica modalità di invio. Per chi formula la Proposta online, le comunicazioni digitali
      sono automaticamente previste; 2) di scegliere che le bollette per i pagamenti relativi al Contratto siano rese disponibili come bollette
      digitali (in formato elettronico anzichè cartaceo) all'indirizzo mail indicato nella Proposta di Contratto secondo le modalità descritte
      all'  Art.  5 delle  CGC.  In  caso  di  problematiche  tecniche  relative  all'invio  in  formato  elettronico,  il fornitore  invierà  al  Cliente  tali
      comunicazioni all'indirizzo fisico. """,
            """ Comunicazioni e bolletta digitale
      Il  Cliente  accetta:  1)  di  ricevere  la  documentazione  contrattuale  (inclusi  le  informazioni  precontrattuali  e il  Contratto)  e le
      comunicazioni relative al Contratto in formato elettronico all'indirizzo email indicato nel Modulo, fatto salvo successivi aggiornamenti
      comunicati al fornitore da parte del Cliente. Tale consenso del Cliente vale anche per tutte le successive comunicazioni per le quali la
      legge o il Contratto non preveda una diversa e specifica modalità di invio. Per chi formula la Proposta online, le comunicazioni digitali
      sono automaticamente previste; 2) di scegliere che le bollette per i pagamenti relativi al Contratto siano rese disponibili come bollette
      digitali (in formato elettronico anzichè cartaceo) all'indirizzo mail indicato nella Proposta di Contratto secondo le modalità descritte
      all'  Art.  5 delle  CGC.  In  caso  di  problematiche  tecniche  relative  all'invio  in  formato  elettronico,  il fornitore  invierà  al  Cliente  tali
      comunicazioni all'indirizzo fisico. """
        ],
        [
            """ * DATO OBBLIGATORIO, UNITAMENTE ALLA COPIA DEL DOCUMENTO DI RICONOSCIMENTO, PER LA STIPULA E/O L'ESECUZIONE DEL CONTRATTO SALVO DIVERSA VALUTAZIONE DEL FORNITORE """,
            """ * DATO OBBLIGATORIO, UNITAMENTE ALLA COPIA DEL DOCUMENTO DI RICONOSCIMENTO, PER LA STIPULA E/O L'ESECUZIONE DEL CONTRATTO SALVO DIVERSA VALUTAZIONE DEL FORNITORE """
        ],
        [
            """** Il dato potrebbe non essere disponibile in fase di sottoscrizione, ma verrà indicato in successive comunicazioni e comunque nella prima fattura
      L'indirizzo email sarà utilizzato dal fornitore per effetuare la pre-registrazione all'Area Personale.""",
            """** Il dato potrebbe non essere disponibile in fase di sottoscrizione, ma verrà indicato in successive comunicazioni e comunque nella prima fattura
      L'indirizzo email sarà utilizzato dal fornitore per effetuare la pre-registrazione all'Area Personale."""
        ],
        [
            "Proposta di contratto per la casa",
            "Proposta di contratto per la casa"
        ],
        [
            "Codice Convenzione\n Codice incaricato \nN° Plico",
            "Codice Convenzione\n Codice incaricato \nN° Plico"
        ]
    ],
    [
        [
            """Con  la  formulazione  della  proposta  di  contratto,  il  Cliente  dichiara  di  aver  preso  visione  dell'informativa  sul  trattamento  dei  dati
      personali, riportata nel Contratto. Barrando le successive caselle, il Cliente può decidere liberamente di esprimere o meno il consenso aL fornitore per l'utilizzo dei dati personali finalizzato alle seguenti attività:""",
            """Con  la  formulazione  della  proposta  di  contratto,  il  Cliente  dichiara  di  aver  preso  visione  dell'informativa  sul  trattamento  dei  dati
      personali, riportata nel Contratto. Barrando le successive caselle, il Cliente può decidere liberamente di esprimere o meno il consenso al fornitore per l'utilizzo dei dati personali finalizzato alle seguenti attività:"""
        ],
        [
            "future iniziative promozionali curate dal fornitore              si no",
            "future iniziative promozionali curate dal fornitore              si no"
        ],
        [
            "analisi e ricerche di mercato effettuate direttamente dal fornitore o attraverso società terze   si no",
            "analisi e ricerche di mercato effettuate direttamente dal fornitore o attraverso società terze   si no"
        ],
        [
            """future iniziative promozionali curate da altre imprese           si no""",
            """future iniziative promozionali curate da altre imprese           si no"""
        ],
        [
            "Proposta di contratto per la casa",
            "Proposta di contratto per la casa"
        ],
        [
            "Codice Convenzione\n Codice incaricato \nN° Plico",
            "Codice Convenzione\n Codice incaricato \nN° Plico"
        ],
    ],
    [
        [
            """ ll Cliente con la sottoscrizione in calce formula la Proposta di Contratto al fornitore e contestualmente dichiara di aver ricevuto, letto e
      accettato i seguenti allegati: ; Condizioni Economiche e gli eventuali relativi allegati; Livelli di qualità
      commerciale  dei  servizi  di  vendita  di  energia  elettrica  e  gas  naturale  ai  sensi  del  TIQV;  Schede  Sintetiche.  Il  termine  entro  cui
      sottoscrivere il presente Modulo è indicato nell' Allegato “Condizioni Economiche”.
      Con la sottoscrizione del Modulo, il Cliente conferma le scelte indicate nel Contratto, si assume la responsabilità della veridicità dei
      dati ivi riportati, ivi incluse le dichiarazioni di cui alle Condizioni Economiche, con particolare riferimento per l'energia elettrica alla
      tipologia d'uso; inoltre dichiara di conoscere le condizioni contrattuali ed economiche del Servizio di Tutela della Vulnerabilità
      Gas  definite  da  ARERA  e  quelle  delle  Offerte  PLACET  del fornitore,  che  può  sempre  scegliere    in  caso  di  richiesta  di
      somministrazione di  energia  elettrica, di  conferire al fornitore i  mandati di  cui all'Art. 4 delle CGC.  Inoltre, con  la  sottoscrizione  del
      Modulo, il Cliente dichiara di conoscere la facoltà del fornitore di recedere dal Contratto, prima dell'esecuzione dello stesso, ai sensi di
      quanto previsto all'Art. 3 delle CGC""",
            """ ll Cliente con la sottoscrizione in calce formula la Proposta di Contratto al fornitore e contestualmente dichiara di aver ricevuto, letto e
      accettato i seguenti allegati: ; Condizioni Economiche e gli eventuali relativi allegati; Livelli di qualità
      commerciale  dei  servizi  di  vendita  di  energia  elettrica  e  gas  naturale  ai  sensi  del  TIQV;  Schede  Sintetiche.  Il  termine  entro  cui
      sottoscrivere il presente Modulo è indicato nell' Allegato “Condizioni Economiche”.
      Con la sottoscrizione del Modulo, il Cliente conferma le scelte indicate nel Contratto, si assume la responsabilità della veridicità dei
      dati ivi riportati, ivi incluse le dichiarazioni di cui alle Condizioni Economiche, con particolare riferimento per l'energia elettrica alla
      tipologia d'uso; inoltre dichiara di conoscere le condizioni contrattuali ed economiche del Servizio di Tutela della Vulnerabilità
      Gas  definite  da  ARERA  e  quelle  delle  Offerte  PLACET  del fornitore,  che  può  sempre  scegliere    in  caso  di  richiesta  di
      somministrazione di  energia  elettrica, di  conferire al fornitore i  mandati di  cui all'Art. 4 delle CGC.  Inoltre, con  la  sottoscrizione  del
      Modulo, il Cliente dichiara di conoscere la facoltà del fornitore di recedere dal Contratto, prima dell'esecuzione dello stesso, ai sensi di
      quanto previsto all'Art. 3 delle CGC""",
        ],
        ["""Data      /     /         Firma del Cliente"""
         """Data      /     /         Firma del Cliente"""],
        [
            """ Il Cliente dichiara di approvare le Condizioni Economiche (riportate nelle CGC e nell'apposito Allegato) e, ai sensi e per gli effetti degli
      artt.  1341  e  1342  del  Codice  Civile,  dichiara  di  aver  preso  visione  e  di  approvare  specificamente  i  seguenti  articoli  delle  CGC:  Art.3
      (mancata  esecuzione,  recesso  prima  dell'esecuzione);  Art.6  (responsabilità  del  Cliente  per  manomissione  del  Contatore  Gas  e/o  del
      Contatore  Elettrico);  Art.8.3  (azioni  in  caso  di  mancato  o  parziale  pagamento);Art.10  (diritto  di  recesso;  cessazione  della
      somministrazione);Art.13 (limitazioni al diritto di rimborso in favore del Cliente).""",
            """ Il Cliente dichiara di approvare le Condizioni Economiche (riportate nelle CGC e nell'apposito Allegato) e, ai sensi e per gli effetti degli
      artt.  1341  e  1342  del  Codice  Civile,  dichiara  di  aver  preso  visione  e  di  approvare  specificamente  i  seguenti  articoli  delle  CGC:  Art.3
      (mancata  esecuzione,  recesso  prima  dell'esecuzione);  Art.6  (responsabilità  del  Cliente  per  manomissione  del  Contatore  Gas  e/o  del
      Contatore  Elettrico);  Art.8.3  (azioni  in  caso  di  mancato  o  parziale  pagamento);Art.10  (diritto  di  recesso;  cessazione  della
      somministrazione);Art.13 (limitazioni al diritto di rimborso in favore del Cliente)."""
        ],
        ["""Data      /     /         Firma del Cliente""",
         """Data      /     /         Firma del Cliente"""],
        [
            """ Il Cliente, consapevole delle sanzioni penali, in caso di dichiarazioni non veritiere, di formazione o uso di atti falsi, richiamate dall'art.
      76  del  D.P.R.  del  28  dicembre  2000,  n.  445,  dichiara,  ai  sensi  e  per  gli  effetti  della  normativa  applicabile,  relativamente  all'unità
      immobiliare sita all'indirizzo di fornitura sopra indicato per la quale si chiede la somministrazione di cui sopra, di essere:
                   Proprietario Conduttore   Detentore ad altro titolo (ad esempio, comodato) """,
            """ Il Cliente, consapevole delle sanzioni penali, in caso di dichiarazioni non veritiere, di formazione o uso di atti falsi, richiamate dall'art.
      76  del  D.P.R.  del  28  dicembre  2000,  n.  445,  dichiara,  ai  sensi  e  per  gli  effetti  della  normativa  applicabile,  relativamente  all'unità
      immobiliare sita all'indirizzo di fornitura sopra indicato per la quale si chiede la somministrazione di cui sopra, di essere:
                   Proprietario Conduttore   Detentore ad altro titolo (ad esempio, comodato) """
        ],
        ["""Data      /     /         Firma del Cliente""",
         """Data      /     /         Firma del Cliente"""]
    ],
    [
        [
            """Nome e Cognome (T  itolare del conto )*      Codice fiscale*
      Coordinate Bancarie o Postali*
      Il  titolare  del  conto  corrente  (“Titolare”)  autorizza  Capogruppo S.p.A..  (nome  dell'effettivo  creditore  Societa S.p.A. -
      codice identificativo  dell'effettivo  creditore  000) a  disporre  addebiti  in  via  continuativa  sul  conto  corrente  identificato  dal
      codice IBAN sopra indicato relativi alle fatture di cui all'Art. 7 delle CGC e la banca ad eseguire l'addebito secondo le disposizioni.
      In caso di mancato pagamento, esso dovrà essere effettuato direttamente a cura del Cliente. Il rapporto con la banca è regolato
      dal contratto stipulato dal debitore con la banca stessa. Il Cliente ha facoltà di richiedere alla banca il rimborso di quanto addebitato,
      secondo  quanto  previsto  nel  suddetto  contratto;  eventuali  richieste  di  rimborso  devono  essere  presentate  entro  e non  oltre  8
      settimane a decorrere  dalla data di  addebito in conto.  I diritti  del debitore riguardanti l'autorizzazione sopra  riportata sono indicati
      nella documentazione  ottenibile  dalla banca. Ulteriori  dettagli operativi relativi  alla domiciliazione  bancaria  o postale delle  bollette
      sono riportati all'Art. 8.1 delle CGC.
         Firma del titolare del C/C (PER RICHIESTA ADDEBITO IN CONTO)""",
            """Nome e Cognome (T  itolare del conto )*      Codice fiscale*
      Coordinate Bancarie o Postali*
      Il  titolare  del  conto  corrente  (“Titolare”)  autorizza  Capogruppo S.p.A..  (nome  dell'effettivo  creditore  Societa S.p.A. -
      codice identificativo  dell'effettivo  creditore  000) a  disporre  addebiti  in  via  continuativa  sul  conto  corrente  identificato  dal
      codice IBAN sopra indicato relativi alle fatture di cui all'Art. 7 delle CGC e la banca ad eseguire l'addebito secondo le disposizioni.
      In caso di mancato pagamento, esso dovrà essere effettuato direttamente a cura del Cliente. Il rapporto con la banca è regolato
      dal contratto stipulato dal debitore con la banca stessa. Il Cliente ha facoltà di richiedere alla banca il rimborso di quanto addebitato,
      secondo  quanto  previsto  nel  suddetto  contratto;  eventuali  richieste  di  rimborso  devono  essere  presentate  entro  e non  oltre  8
      settimane a decorrere  dalla data di  addebito in conto.  I diritti  del debitore riguardanti l'autorizzazione sopra  riportata sono indicati
      nella documentazione  ottenibile  dalla banca. Ulteriori  dettagli operativi relativi  alla domiciliazione  bancaria  o postale delle  bollette
      sono riportati all'Art. 8.1 delle CGC.
         Firma del titolare del C/C (PER RICHIESTA ADDEBITO IN CONTO)"""
        ]
    ],
    [
        ["attivazione", "attivazione"],
        [
            """ Il  Cliente,  come  di  seguito  identificato,  propone  a Societa S.p.A. con  sede  legale  in città  (XX),  Via  Vai 0; Capitale  Sociale  Euro i.v.,  Registro  Imprese  di  Citta  Codice  Fiscale  e Partita  Iva 0,  R.E.A.  Citta  n.0,  società  soggetta  all'attività  di  direzione  e coordinamento  di  Capogruppo S.p.A.,  di concludere un contratto di somministrazione di gas naturale  (di seguito “Contratto”), secondo i termini e le condizioni
      indicate nel presente modulo contrattuale (di seguito “Modulo ”) e nei relativi allegati, che costituiscono parte integrante e sostanziale
      del Contratto.""",
            """ Il  Cliente,  come  di  seguito  identificato,  propone  a Societa S.p.A. con  sede  legale  in  città  (XX),  Via  Vai 0; Capitale  Sociale  Euro i.v.,  Registro  Imprese  di  Citta  Codice  Fiscale  e Partita  Iva 0,  R.E.A.  Citta  n.0,  società  soggetta  all'attività  di  direzione  e coordinamento  di  Capogruppo S.p.A.,  di concludere un contratto di somministrazione di gas naturale  (di seguito “Contratto”), secondo i termini e le condizioni
      indicate nel presente modulo contrattuale (di seguito “Modulo ”) e nei relativi allegati, che costituiscono parte integrante e sostanziale
      del Contratto."""
        ],
    ],
]
df_sorted = df_training.where(F.col("flg_serving") == True).orderBy("chunk")
df = df_sorted.withColumn(
    "num_id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
)
labels_udf = F.udf(
    lambda indx: ground_truth[indx - 1], T.ArrayType(T.ArrayType(T.StringType()))
)
df_target = df.withColumn("ground_truth", labels_udf("num_id"))
# COMMAND ----------
extra_column=('ground_truth',
T.ArrayType(
            T.ArrayType(T.StringType())))
cols_list = list(cols.keys())[1:15]
schema = build_schema({k: cols[k] for k in cols_list}, extra_col=extra_column)
df = df_target.selectExpr(*[f"{col} as {col}" for col in schema.fieldNames()])
# Save the processed DataFrame to the specified output table
df.write.option("overwriteSchema", "true").mode("overwrite").format("delta").saveAsTable(output_path)
