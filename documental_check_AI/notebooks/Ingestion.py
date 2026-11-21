# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # Document Ingestion Pipeline
# MAGIC ## This notebook handles the ingestion of campaign documents (CE, CGC, SR) by comparing current and past campaign folders and creating a master table with file correspondences.

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
from src.utils.logger import get_logger
from datetime import datetime
from src.config.config import Config
from src.master.master_ingestion import get_corrisponding_files, get_corrisponding_plichi
from src.utils.utils import get_campaign_dates, get_campaign_folder_path, build_schema
import os
import re
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

# COMMAND ----------

# DBTITLE 1,Initialize logger and Spark session
# Initialize logging and Spark
logger = get_logger()
spark: SparkSession = SparkSession.builder.getOrCreate()
logger_py4j = logging.getLogger("py4j")
logger_py4j.setLevel(logging.ERROR)  # Reduce py4j verbosity

# COMMAND ----------

# DBTITLE 1,Load configuration
# Load application configuration from YAML
config = Config()
config.read_path("../src/config/config.yml")

# Extract configuration values
catalog = config.get_config_value("ingestion.catalog")
schema = config.get_config_value("ingestion.schema")
volume = config.get_config_value("ingestion.volume")
output_table = config.get_config_value("ingestion.table_ingestion")
cols = config.get_config_value("ingestion.columns")
partition_columns = config.get_config_value("ingestion.partition_columns")

# Build paths
volume_directory = f"/Volumes/{catalog}/{schema}/{volume}/Campagne/"
output_path = f"{catalog}.{schema}.{output_table}"
# COMMAND ----------

# DBTITLE 1,Get campaign dates
# Automatically determine current and past campaign dates
currentcampaign_month_year, pastcampaign_month_year = get_campaign_dates(volume_directory)
print("Current campaign:", currentcampaign_month_year)
print("Past campaign:", pastcampaign_month_year)

# COMMAND ----------

# DBTITLE 1,Get campaign folder paths
# Find and validate campaign folder paths
pastcampaign_paths = get_campaign_folder_path(volume_directory, pastcampaign_month_year)
currentcampaign_paths = get_campaign_folder_path(volume_directory, currentcampaign_month_year)

# COMMAND ----------

# DBTITLE 1,Process campaign documents
# Validate current campaign paths and process documents
if currentcampaign_paths == '"No campaign folder found"':
    exit("No campaign folders found for the current month")
elif len(currentcampaign_paths) == 0:
    exit("Error during unzipping the campaign folders for the current month")
else:
    # Process CE (Condizioni Economiche) documents
    CE_rows = get_corrisponding_files(
        macrosegmenti=["RESIDENZIALI", "MICROBUSINESS", "AUP"],
        file_type="CE",
        currentcampaign_paths=currentcampaign_paths,
        pastcampaign_paths=pastcampaign_paths,
    )

    # Process CGC (Condizioni Generali di Contratto) documents
    CGC_rows = get_corrisponding_files(
        macrosegmenti=["RESIDENZIALI", "MICROBUSINESS", "AUP"],
        file_type="CGC",
        currentcampaign_paths=currentcampaign_paths,
        pastcampaign_paths=pastcampaign_paths,
    )

    # Process SR (Scheda di Riepilogo) documents
    SR_rows = get_corrisponding_files(
        macrosegmenti=["RESIDENZIALI"],
        file_type="SR",
        currentcampaign_paths=currentcampaign_paths,
        pastcampaign_paths=pastcampaign_paths,
    )
    # Process SS (Scheda Sintetica) documents
    SS_rows = get_corrisponding_files(
        macrosegmenti=["RESIDENZIALI", "MICROBUSINESS"],
        file_type="SS",
        currentcampaign_paths=currentcampaign_paths,
        pastcampaign_paths=pastcampaign_paths,
    )
    PLICHI_rows = get_corrisponding_plichi(
        macrosegmenti=["RESIDENZIALI", "MICROBUSINESS", "AUP"],
        file_type="PLICHI",
        currentcampaign_paths=currentcampaign_paths,
        pastcampaign_paths=pastcampaign_paths,
    )

# COMMAND ----------

# DBTITLE 1,Combine all document rows
# Combine all processed document rows
rows = CE_rows + CGC_rows + SR_rows + SS_rows + PLICHI_rows

# COMMAND ----------

# DBTITLE 1,Create and save DataFrame
# Build schema and create DataFrame
cols_list = list(cols.keys())
schema = build_schema({k: cols[k] for k in cols_list})
df = spark.createDataFrame(rows, schema)

# Add snapshot date
df = df.withColumn("DATA_SNAPSHOT", F.current_date())

# Save to Delta table
df.write.format("delta").mode("overwrite").partitionBy(partition_columns[0], partition_columns[1]).option(
    "replaceWhere", "current_campaign LIKE '%" + currentcampaign_month_year + "%'"
).saveAsTable(output_path)
