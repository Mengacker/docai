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
from src.config import Config
import os
from src.utils.logger import get_logger
import logging
import mlflow
from src.metrics.extraction_module import validate_answer_V2, scorers
from mlflow.genai.scorers import Guidelines, scorer
from mlflow.genai import evaluate
import mlflow.genai
import string
import difflib

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
input_table = config.get_config_value("master_creation.CE.output_table_train")
output_table = config.get_config_value("master_creation.CE.output_table_train")
cols = config.get_config_value("master_creation.CE.columns")
volume_directory = f"/Volumes/{catalog}/{schema}/{volume}"
# COMMAND ----------
input_path = f"{catalog}.{schema}.{input_table}"
output_path = f"{catalog}.{schema}.{output_table}"
# COMMAND ----------

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
    adapter = dspy.TwoStepAdapter(lm)
    return dspy.settings.configure(lm=lm, adapter=adapter)
# COMMAND ----------

# DBTITLE 1, Load training data

df_training = spark.table(input_path)
# COMMAND ----------

# DBTITLE 1, Extraction_module
experiment_name = "Extraction_module"
experiment_path = "/Workspace/Shared/DataScience/Check_Doc/Extraction_model/CE"
experiment_id = create_or_get_experiment(experiment_name, experiment_path)
# mlflow.dspy.autolog()
mlflow.set_experiment(experiment_id=experiment_id)
mlflow.set_tracking_uri("databricks")
#COMMAND ----------
model_name = "check_documentale_gpt4o"
configure_dspy_model(model_name)
# COMMAND ----------

class ExtractionSignature(dspy.Signature):
    """Estrai tutti i paragrafi dai seguenti due documenti in italiano. Restituisci una lista di liste, ogni lista contiene i 2 paragrafi corrispondenti ciascuno proveniente dai corrispettivi documenti. IGNORA le differenze di spazi multipli, tabulazioni e rientri. Se un paragrafo è presente solo in uno dei due documenti, inserisci il paragrafo disponibile e una stringa vuota ('') al posto del paragrafo mancante."""
    question: str = dspy.InputField()
    answer: list[list[str]] = dspy.OutputField(
        desc="Lista di liste, ogni lista contiene i 2 paragrafi corrispondenti dai due documenti in italiano."
    )

extractor = dspy.Predict(ExtractionSignature)

# COMMAND ----------

x = [row.extracted_chunk_clean for row in df_training.collect()]
y = [row.ground_truth for row in df_training.collect()]
trainset = [
    dspy.Example(question=document, answer=truth).with_inputs("question")
    for document, truth in zip(x, y)
]

# COMMAND ----------

# DBTITLE 1, Evaluation before training
mlflow.dspy.autolog(log_traces_from_eval=True)
with mlflow.start_run(run_name="Extarction_Model_Before_Compilation_CE"):
    #evaluator = dspy.evaluate.Evaluate(
    #    devset=trainset,
    #    return_all_scores=True,
    #    show_progress=True,
    #)
    data = [
    {
        "inputs": {
            "documento": document
        },
        "outputs": {
            "risposta": extractor(question=document).answer
        }
        ,
        "expectations": {
            "ground_truth": ground_truth
        }
    }
    for document, ground_truth in zip(x, y)
    ]
    #results_match= evaluator(bfs_optimized, metric=validate_answer_V2)

    # Extract data from the result - results is a list of tuples (example, prediction, score)
    #outputs = [r[1] for r in results_match.results]  # r[1] is the prediction
    results_guidelines = mlflow.genai.evaluate(
    data=data,
    scorers=scorers)
    #all_scores = [r[2] for r in results_match.results]  # r[2] is the score
    #aggregated_score = sum(all_scores) / len(all_scores) if all_scores else 0
    # Log the aggregated score
    #mlflow.log_metric("similarity_match", results_guidelines.metrics['validate_answer_V2/mean'])
    #mlflow.genai.register_prompt(name="p_sbx_data.data.extractionmodel_pc", template=f"{prompt}")
    # Log the detailed evaluation results as a table
    mlflow.log_table(
        {
            "document": [x.question for x in trainset],
            "truth": [x.answer for x in trainset],
            #"output": outputs,
            #"score": all_scores,
        },
        artifact_file="eval_results.json",
    )
# COMMAND ----------
# Enabling tracing for DSPy evaluation
mlflow.dspy.autolog(log_traces_from_eval=True)
with mlflow.start_run(run_name="Extarction_Model_Before_Compilation_CE"):
    evaluator = dspy.evaluate.Evaluate(
        devset=trainset,
        return_all_scores=True,
        show_progress=True,
    )
    result = evaluator(extractor, metric=validate_answer_V2)

    # Extract data from the result - results is a list of tuples (example, prediction, score)
    outputs = [r[1] for r in result.results]  # r[1] is the prediction
    all_scores = [r[2] for r in result.results]  # r[2] is the score
    aggregated_score = sum(all_scores) / len(all_scores) if all_scores else 0

    # Log the aggregated score
    mlflow.log_metric("exact_match", aggregated_score)
    #mlflow.genai.register_prompt(name="p_sbx_data.data.extractionmodel_CE", template=f"{prompt}")
    # Log the detailed evaluation results as a table
    mlflow.log_table(
        {
            "document": [x.question for x in trainset],
            "truth": [x.answer for x in trainset],
            "output": outputs,
            "score": all_scores,
        },
        artifact_file="eval_results.json",
    )
#COMMAND ----------
mlflow.dspy.autolog(log_traces_from_compile=True)
teleprompter = dspy.BootstrapFewShot(metric=validate_answer_V2)
bfs_optimized = teleprompter.compile(extractor, trainset=trainset)
# COMMAND ----------
# Enabling tracing for DSPy evaluation
mlflow.dspy.autolog(log_traces_from_eval=True)
with mlflow.start_run(run_name="Extarction_Model_After_Compilation_PC"):
    #evaluator = dspy.evaluate.Evaluate(
    #    devset=trainset,
    #    return_all_scores=True,
    #    show_progress=True,
    #)
    data = [
    {
        "inputs": {
            "documento": document
        },
        "outputs": {
            "risposta": bfs_optimized(question=document).answer
        }
        ,
        "expectations": {
            "ground_truth": ground_truth
        }
    }
    for document, ground_truth in zip(x, y)
    ]
    #results_match= evaluator(bfs_optimized, metric=validate_answer_V2)

    # Extract data from the result - results is a list of tuples (example, prediction, score)
    #outputs = [r[1] for r in results_match.results]  # r[1] is the prediction
    results_guidelines = mlflow.genai.evaluate(
    data=data,
    scorers=scorers)
    #all_scores = [r[2] for r in results_match.results]  # r[2] is the score
    #aggregated_score = sum(all_scores) / len(all_scores) if all_scores else 0
    # Log the aggregated score
    mlflow.log_metric("similarity_match", results_guidelines.metrics['validate_answer_V2/mean'])
    #mlflow.genai.register_prompt(name="p_sbx_data.data.extractionmodel_pc", template=f"{prompt}")
    # Log the detailed evaluation results as a table
    mlflow.log_table(
        {
            "document": [x.question for x in trainset],
            "truth": [x.answer for x in trainset],
            #"output": outputs,
            #"score": all_scores,
        },
        artifact_file="eval_results.json",
    )

# COMMAND ----------

# DBTITLE 1, Save the model
with mlflow.start_run(run_name="log model"):
    model_info = mlflow.dspy.log_model(
        bfs_optimized,
        name="ExtractorModel_PC",
        input_example="documents",
    )

# COMMAND ----------

model_uri = model_info.model_uri
model_uri
