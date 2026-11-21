# Databricks notebook source
# MAGIC %md
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
from atum_ndp.config import Config
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
from src.metrics.extraction_module import validate_answer_V2

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
input_table = config.get_config_value("master_creation.SR.output_table_train")
output_table = config.get_config_value("master_creation.SR.output_table_train")
# cols = config.get_config_value("master_creation.SR.output_fields")
volume_directory = f"/Volumes/{catalog}/{schema}/{volume}"

# COMMAND ----------

# DBTITLE 1, get dates
# currentcampaign_year = datetime.now().year
# currentcampaign_month = datetime.now().month
# if currentcampaign_month == 1:
#    pastcampaign_month = 12
# else:
#    pastcampaign_month = currentcampaign_month - 1
# pastcampaign_month_year = f"{pastcampaign_month:02}{currentcampaign_year:04}"
# currentcampaign_month_year = f"{currentcampaign_month:02}{currentcampaign_year:04}"
pastcampaign_month_year = "072025"
currentcampaign_month_year = "082025"

# COMMAND ----------

input_path = f"{catalog}.{schema}.{input_table}_{pastcampaign_month_year}_{currentcampaign_month_year}"
output_path = f"{catalog}.{schema}.{output_table}_{pastcampaign_month_year}_{currentcampaign_month_year}"

# COMMAND ----------

# DBTITLE 1, Import custom functions
def create_or_get_experiment(experiment_name, experiment_path):
    """
    Create or retrieve an MLflow experiment by name and set the artifact storage path.

    Parameters:
    - experiment_name (str): The name of the experiment.
    - experiment_path (str): The artifact location path where the experiment will store results.

    Returns:
    - str: The experiment ID.
    """
    experiment_name = experiment_path.replace("/Workspace", "") + "/" + experiment_name
    try:
        # Try to create a new experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
    except Exception:
        # If it already exists, retrieve the experiment ID
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"Experiment '{experiment_name}' already exists with ID: {experiment_id}")

    return experiment_id


def configure_dspy_model(model_name):
    lm = dspy.LM(
        model=f"databricks/{model_name}",
        max_tokens=16384,
        temperature=0,
    )
    return dspy.settings.configure(lm=lm)

# COMMAND ----------

# DBTITLE 1, Load training data

df_training = spark.table(input_path)

# COMMAND ----------

# DBTITLE 1, Extraction_module
experiment_name = "Extraction_module"
experiment_path = "/Workspace/Shared/DataScience/Check_Doc/SR"
experiment_id = create_or_get_experiment(experiment_name, experiment_path)
# mlflow.dspy.autolog()
mlflow.set_experiment(experiment_id=experiment_id)
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

# DBTITLE 1, model loading
model_name = "gpt-4o-ptu-batch"
model = configure_dspy_model(model_name)


class ExtractionSignature(dspy.Signature):
    documents: str = dspy.InputField()
    corresponding_contractual_paragraphs: list[list[str]] = dspy.OutputField(
        desc="Extract all the paragraphs in the two documents. Return a list of lists, each list contains the corresponding paragraphs from the two documents. When a paragraph is present just one of the two documents, return a list with the paragraph and an empty string as the other document's paragraph."
    )


class ExtractionProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(ExtractionSignature)

    def forward(self, documents: str):
        return self.extractor(documents=documents).corresponding_contractual_paragraphs


# Instantiate the program once (important for Spark workers)
extraction_program = ExtractionProgram()

# COMMAND ----------

x = [row.extracted_chunk_clean for row in df_training.collect()]
y = [row.ground_truth for row in df_training.collect()]
trainset = [
    dspy.Example(documents=document, corresponding_contractual_paragraphs=truth).with_inputs("documents")
    for document, truth in zip(x, y)
]

# COMMAND ----------

# DBTITLE 1, Training

scores = []
for x in trainset:
    pred = extraction_program.forward(documents=x.inputs().documents)
    score = validate_answer_V2(pred, x.corresponding_contractual_paragraphs)
    scores.append(score)
scores

# COMMAND ----------

mlflow.dspy.autolog(log_traces_from_compile=True)
teleprompter = dspy.BootstrapFewShot(metric=validate_answer_V2)
bfs_optimized = teleprompter.compile(extraction_program, trainset=trainset)

# COMMAND ----------

# Enabling tracing for DSPy evaluation
mlflow.dspy.autolog(log_traces_from_eval=True)
with mlflow.start_run(run_name="Extarction_eval_after_compilation"):
    scores = []
    outputs = []
    for x in trainset:
        pred = bfs_optimized.forward(documents=x.inputs().documents)
        score = validate_answer_V2(pred, x.corresponding_contractual_paragraphs)
        outputs.append(pred)
        scores.append(score)
    all_scores = scores
    aggregated_score = sum(scores) / len(scores)
    # Log the aggregated score
    mlflow.log_metric("exact_match", aggregated_score)
    # Log the detailed evaluation results as a table
    mlflow.log_table(
        {
            "document": [x.documents for x in trainset],
            "truth": [y.corresponding_contractual_paragraphs for y in trainset],
            "output": [z for z in outputs],
            "score": [score for score in all_scores],
        },
        artifact_file="eval_results.json",
    )

# COMMAND ----------

# DBTITLE 1, Save the model
with mlflow.start_run(run_name="log model"):
    model_info = mlflow.dspy.log_model(
        bfs_optimized,
        name="model",
        input_example="documents",
    )

# COMMAND ----------

model_uri = model_info.model_uri
model_uri
