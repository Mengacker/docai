"""Functions to postprocess predictions: join, assign percentiles, and save results."""

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F

from src.config import Config
from src.serving.utils import (
    compute_percentile,
    get_delta_between_bucktes,
    add_key_clientd365,
    get_customer_base,
    get_inference_results,
    ensure_columns_and_cast,
)
from src.utils.logger import get_logger

logger = get_logger()
spark = SparkSession.builder.getOrCreate()

# Globals config
config = Config()
config.read_path("../src/config/config.yml")
INFERENCE_RESULTS_CONF = config.get_config_value("inference_results")
PROPENSITY_OUTPUT_CONF = config.get_config_value("propensity_output")
CB_PATH = config.get_config_value("customer_base.path")


def full_post_inference_pipeline(
    snapshot_date: str,
    prediction_column: str = "PRED_PROBA",
    id_column: str = "PK_PDF",
    new_prediction_column: str = "CHURN_SCORE_PDF",
    percentile_column: str = "CHURN_PERC_PDF",
    fornitura_column_in: str = "PK_ASSET_LINE_ITEM",
    fornitura_column_out: str = "COD_FORNITURA",
    snapshot_date_col_in: str = "DATA_SNAPSHOT",
    snapshot_date_col_out: str = "CHURN_RUN_DATE",
    save_table: bool = True,
) -> DataFrame:
    """
    Full post inference processing pipeline:
    - Loads config and data
    - Computes percentiles and quality metrics
    - Joins with customer base
    - Adds required columns and casts types
    - Optionally saves the result table

    Returns:
        Final processed DataFrame.
    """
    # Load and process CB and inference results
    cb = get_customer_base(
        CB_PATH, snapshot_date, id_column, fornitura_column_in, fornitura_column_out
    )
    inference_results = get_inference_results(
        INFERENCE_RESULTS_CONF,
        snapshot_date,
        id_column,
        prediction_column,
        snapshot_date_col_in,
        snapshot_date_col_out,
        new_prediction_column,
    )

    # Compute percentiles and assess quality of buckets
    inference_results = compute_percentile(
        inference_results,
        in_col_name=new_prediction_column,
        out_col_name=percentile_column,
        swap_cols=False,
        ascending=False,
    )
    delta = get_delta_between_bucktes(inference_results, new_prediction_column)
    logger.info("Delta between less and most populated buckets is %s", delta)

    # Add customer base data
    inference_results = inference_results.join(cb, id_column, "inner")

    # Get max score and min perc by customer (results are for PDF)
    df_cli = inference_results.groupby("PK_ACCOUNT").agg(
        F.min(percentile_column).alias("CHURN_MIN_PERC_CLI"),
        F.max(new_prediction_column).alias("CHURN_MAX_SCO_CLI"),
    )
    inference_results = inference_results.join(df_cli, on="PK_ACCOUNT", how="left")

    # Add key clientd365
    inference_results = add_key_clientd365(inference_results, snapshot_date)

    # Ensure all required columns and cast types
    inference_results = ensure_columns_and_cast(
        inference_results, PROPENSITY_OUTPUT_CONF["must_be_columns"]
    )

    # Select columns in required order
    inference_results = inference_results.select(
        *PROPENSITY_OUTPUT_CONF["must_be_columns"].keys()
    )

    # Save if required
    if save_table:
        output_path = f"{PROPENSITY_OUTPUT_CONF['catalog']}.{PROPENSITY_OUTPUT_CONF['schema']}.{PROPENSITY_OUTPUT_CONF['table_name']}"
        inference_results.write.partitionBy(
            [PROPENSITY_OUTPUT_CONF["partition_column"]]
        ).mode("overwrite").option("partitionOverwriteMode", "dynamic").saveAsTable(
            output_path
        )
        logger.info("Final propensity output correctly saved in '%s'", output_path)

    return inference_results
