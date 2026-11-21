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
## MAGIC %pip install -r /Workspace/.../src/requirements.txt
## MAGIC dbutils.library.restartPython()

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
import dspy
from pyspark.sql import SparkSession
from src.config import Config
import os
import re
from datetime import datetime
from pyspark.sql import Row
import pyspark.sql.types as T
import pyspark.sql.functions as F
from src.utils.logger import get_logger
import logging
from typing import Literal
import mlflow
from src.utils.utils import get_campaign_dates, build_schema, retry_with_exponential_backoff
from src.utils.dspy_utils import configure_dspy_model
from src.model.DifferencesSummarizer_module import summary
from src.model.Extraction_module import extractor
from src.model.Comparison_module import check_identical_texts_udf

# COMMAND ----------

# DBTITLE 1, Initialize logger and Spark session
logger = get_logger()
spark: SparkSession = SparkSession.builder.getOrCreate()
logger_py4j = logging.getLogger("py4j")
logger_py4j.setLevel(logging.ERROR)

# COMMAND ----------

# DBTITLE 1, Load configuration
config = Config()
config.read_path("../src/config/config.yml")
catalog = config.get_config_value("master_creation.catalog")
schema = config.get_config_value("master_creation.schema")
volume = config.get_config_value("master_creation.volume")
input_table = config.get_config_value("master_creation.table_inference")
output_table = config.get_config_value("master_creation.table_prediction")
cols = config.get_config_value("master_creation.columns_prediction")
partition_columns = config.get_config_value("master_creation.partition_columns")
volume_directory = f"/Volumes/{catalog}/{schema}/{volume}"
volume_campagne = f"/Volumes/{catalog}/{schema}/{volume}/Campagne"
input_path = f"{catalog}.{schema}.{input_table}"
output_path = f"{catalog}.{schema}.{output_table}"

# COMMAND ----------

# DBTITLE 1, get campaigns dates
currentcampaign_month_year, pastcampaign_month_year = get_campaign_dates(volume_campagne)
print("Current campaign:", currentcampaign_month_year)
print("Past campaign:", pastcampaign_month_year)

# COMMAND ----------

# pastcampaign_month_year = '052025'
# currentcampaign_month_year ='062025'

# COMMAND ----------

input_path = f"{catalog}.{schema}.{input_table}"
output_path = f"{catalog}.{schema}.{output_table}"
output_dir = f"{volume_directory}/Risultato_Campagne_{pastcampaign_month_year}_{currentcampaign_month_year}"
output_file = f"{output_dir}/SS_{pastcampaign_month_year}_{currentcampaign_month_year}.csv"
os.makedirs(output_dir, exist_ok=True)

# COMMAND ----------

df = (
    spark.table(input_path)
    .where(F.col("current_campaign").contains(currentcampaign_month_year))
    .where(F.col("document_type") == "SS")
    .where(F.col("flg_serving") == True)
)
df = df.dropDuplicates()

# COMMAND ----------
model_name = "check_documentale_gpt-5-1"
configure_dspy_model(model_name)
#extractor_module= mlflow.dspy.load_model(model_uri)
# COMMAND ----------

rows = []
cols_list = list(cols.keys())[1:15]

for row in df.select(cols_list).collect():
    row_dict = row.asDict()

    # Wrap the call in a lambda so retry_with_exponential_backoff can call it
    row_dict["corresponding_contractual_paragraphs"] = retry_with_exponential_backoff(
        lambda: extractor(question=row_dict["extracted_chunk_clean"]).answer
    )

    rows.append(Row(**row_dict))

# Build schema and create DataFrame
schema = build_schema({k: cols[k] for k in cols_list + ['corresponding_contractual_paragraphs']})
df_extract_pred = spark.createDataFrame(rows, schema)

# COMMAND ----------

df_compar_pred = df_extract_pred.withColumn(
    "comparison_lists", check_identical_texts_udf(F.col("corresponding_contractual_paragraphs"))
)

# COMMAND ----------

df_compar_pred = df_compar_pred.withColumn(
    "comparison", F.aggregate(F.col("comparison_lists"), F.lit(0), lambda acc, x: acc + x)
)
df_compar_pred.display()


# COMMAND ---------
rows = []
cols_list = list(cols.keys())[1:18]
for row in df_compar_pred.select(cols_list).collect():
    row_dict = row.asDict()
    if row["comparison"] != 0:
        pred =summary(row["corresponding_contractual_paragraphs"], row["comparison_lists"])
        row_dict["summary"] = pred[0]
        row_dict["num_paragraph"] = pred[1]
        row_dict["list_of_differences"] = pred[2]
        rows.append(Row(**row_dict))
    else:
        row_dict["summary"] = ["Nessuna differenza rilevata tra i due paragrafi."] * len(
            row["comparison_lists"]
        )
        row_dict["num_paragraph"] = []
        row_dict["list_of_differences"] = [["Nessuna differenza rilevata tra i due paragrafi."]]* len(
            row["comparison_lists"]
        )
        rows.append(Row(**row_dict))
schema = build_schema({k: cols[k] for k in cols_list + ['summary', 'num_paragraph', 'list_of_differences']})
df_pred_summary = spark.createDataFrame(rows , schema=schema)

# COMMAND ----------

df = df_pred_summary.withColumn("DATA_SNAPSHOT", F.current_date()).select(cols)
df.write.format("delta").mode("overwrite").partitionBy(partition_columns[0], partition_columns[1]).option(
    "replaceWhere", "current_campaign LIKE '%" + currentcampaign_month_year + "%' AND document_type = 'SS'"
).saveAsTable(output_path)

# COMMAND ----------

columns = [
    "macrosegmento",
    "commodity",
    "document_type",
    "current_campaign",
    "past_campaign",
    "current_file_name",
    "past_file_name",
    "chunk_title",
    "comparison",
]

# COMMAND ----------

df_zipped = df_pred_summary.withColumn(
    "zipped", F.arrays_zip("corresponding_contractual_paragraphs", "summary", "num_paragraph", "list_of_differences")
)
df_exploded = df_zipped.select(*columns, F.explode("zipped").alias("zipped")).select(
    *columns,
    F.col("zipped.corresponding_contractual_paragraphs").alias("corresponding_contractual_paragraphs"),
    F.col("zipped.summary").alias("summary"),
    F.col("zipped.num_paragraph").alias("num_paragraph"),
    F.col("zipped.list_of_differences").alias("list_of_differences"),
)
df_exploded.where(F.col("summary") != "Nessuna differenza rilevata tra i due paragrafi.").display()

# COMMAND ----------

df = df_exploded.withColumn(
    "corresponding_contractual_paragraphs",
    F.concat_ws(" \n Past Paragraphs:  ", "corresponding_contractual_paragraphs"),
)

# COMMAND ----------

#(df.write.option("overwriteSchema", "true").mode("overwrite").format("delta").saveAsTable(f"{output_path}_bus"))

# COMMAND ----------

df = df.where(F.col("summary") != "Nessuna differenza rilevata tra i due paragrafi.")
df.toPandas().to_csv(output_file, encoding="utf-8-sig", header=True, index=False, sep=";")
