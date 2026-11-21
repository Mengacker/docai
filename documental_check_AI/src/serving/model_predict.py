"""Function for retrieving registered model and serving it."""

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
import pyspark.sql.types as T
import pandas as pd

from src.config import Config

from src.utils.mlflow_utils import load_deployed_model
from src.utils.logger import get_logger
from src.utils.save_utils import save_uc_table

import ray

from mlflow.utils.databricks_utils import get_databricks_env_vars
from src.utils.ray_utils import config_and_setup_ray_cluster


logger = get_logger()
spark = SparkSession.builder.getOrCreate()

# Global config variables
config = Config()
config.read_path("../src/config/config.yml")
CATALOG: str = config.get_config_value("inference_results.catalog")
SCHEMA: str = config.get_config_value("inference_results.schema")
MASTER_NAME: str = config.get_config_value("inference_results.table_name")
PARTITION_COLUMN: str = config.get_config_value("inference_results.partition_column")
OUTPUT_FIELDS: list = config.get_config_value("inference_results.output_fields")
MODEL_NAME: str = config.get_config_value("mlflow_config.registered_model_name")
TRACKING_URI: str = config.get_config_value("mlflow_config.tracking_uri")
RAY_LOG_PATH: str = config.get_config_value("ray_params.log_path")
RAY_INFERENCE_CONCUNRENCY: int = config.get_config_value(
    "ray_params.inference_concurrency"
)

MLFLOW_DB_CREDS = get_databricks_env_vars("databricks")
DBX_PATH: str = f"{CATALOG}.{SCHEMA}.{MASTER_NAME}"
MODEL_URI: str = f"models:/{MODEL_NAME}@champion"
N_BATCHES: int = RAY_INFERENCE_CONCUNRENCY * 32

TYPE_MAP = {
    "string": T.StringType(),
    "double": T.DoubleType(),
    "bigint": T.LongType(),
    "map": T.MapType(T.StringType(), T.StringType()),
}


def predict_batch(
    master_inference: pd.DataFrame, save_table: bool = False
) -> DataFrame:
    """
    Load the model and predict on master inference.

    Args:
        master_inference (DataFrame): Input DataFrame for inference.
        save_table (bool): Whether to save the results to Unity Catalog.

    Returns:
        DataFrame: DataFrame with predictions and model features.
    """
    # Setup Ray, Mlflow, and load champion model
    config_and_setup_ray_cluster(
        mlflow_db_creds=MLFLOW_DB_CREDS, ray_log_path=RAY_LOG_PATH
    )
    mlflow.set_registry_uri(TRACKING_URI)
    loaded_model = load_deployed_model(MODEL_URI, as_spark_udf=False)
    features_list = [
        name["name"] for name in loaded_model.metadata.signature.inputs.to_dict()
    ]

    # Distributed batch infenrence (it also collapse features in single column)
    logger.info("Start batch inference")
    output_df = (
        ray.data.from_pandas(master_inference, override_num_blocks=N_BATCHES)
        .map_batches(
            _predict,
            fn_kwargs={"model": loaded_model, "features_list": features_list},
            batch_format="pandas",
            concurrency=RAY_INFERENCE_CONCUNRENCY,
        )
        .to_pandas()
    )
    logger.info(
        "Batch inference completed. Shutting down ray clusters, then processing output."
    )
    ray.util.spark.shutdown_ray_cluster()

    # Filter output fields and get back to spark Dataframe with dedicated schema
    schema_fields = [
        T.StructField(col_name, TYPE_MAP[data_type], True)
        for col_name, data_type in OUTPUT_FIELDS.items()
    ]
    output_sdf = spark.createDataFrame(
        output_df[OUTPUT_FIELDS.keys()], schema=T.StructType(schema_fields)
    )

    if save_table:
        save_uc_table(output_sdf, DBX_PATH, partition_column=PARTITION_COLUMN)
    return output_sdf


def _predict(batch, model, features_list):
    """Inner function to use with ray to distribute inference.
    It computes also features dictionary since it is a distributable task.
    """
    batch["PRED_PROBA"] = model.predict(
        batch, params={"predict_method": "predict_proba_target"}
    )
    batch["PRED_BINARY"] = model.predict(batch, params={"predict_method": "predict"})
    batch["MODEL_FEATURES"] = batch[features_list].apply(
        lambda row: {k: str(v) for k, v in row.to_dict().items()}, axis=1
    )
    return batch
