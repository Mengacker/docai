"""Utilities functions for post processing inference results. They are used in post_inference_processing.py."""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window


from src.utils.logger import get_logger

logger = get_logger()
spark = SparkSession.builder.getOrCreate()


def get_delta_between_bucktes(df, col_name):
    """Compute the delta between buckets."""
    cnt_df = df.groupby(col_name).count()
    min_cnt = cnt_df.selectExpr("min(count)").first()[0]
    max_cnt = cnt_df.selectExpr("max(count)").first()[0]
    delta = max_cnt - min_cnt
    return delta


def compute_percentile(
    df: DataFrame,
    in_col_name: str,
    out_col_name: str = None,
    buckets: int = 100,
    ascending: bool = True,
    swap_cols: bool = False,
) -> DataFrame:
    """
    Compute the percentile sorting according to a column of a Spark DataFrame.

    Args:
        df: The input Spark DataFrame.
        in_col_name: The name of the column to sort for.
        out_col_name: The name of the output column. If not provided, a default name will be used.
        buckets: The number of buckets to divide the data into. Default is 100.
        ascending: Whether to sort the data in ascending order. Default is True.
        swap_cols: Whether to swap the names of the input and output columns. Default is False.

    Returns:
        DataFrame: The input DataFrame with an additional column representing the percentile.

    """
    if out_col_name is None:
        out_col_name = "PERC_" + in_col_name

    window = (
        Window.orderBy(in_col_name)
        if ascending
        else Window.orderBy(F.desc(in_col_name))
    )
    out_df = df.withColumn(out_col_name, F.ntile(buckets).over(window).cast("double"))

    if swap_cols:
        out_df = out_df.select(
            *list(set(out_df.columns) - {in_col_name, out_col_name}),
            F.col(out_col_name).alias(in_col_name),
            F.col(in_col_name).alias(out_col_name),
        )
    return out_df


def add_key_clientd365(df_out, snapshot_date):
    """
    Adds `COD_CLIENTE` key to `df_out` based on the client's migration status in Dynamics 365.
    Necessary for reach the CAR.

    Args:
        df_out (DataFrame): The input dataframe.
        snapshot_date (str): The snapshot date for retrieving the migration status.

    Returns:
        DataFrame: The modified dataframe with the added key.
    """
    from plenitude_metrics.account import MetricaFlgMigratoD365

    logger.info("Adding COD_CLIENTE key")
    df_map = MetricaFlgMigratoD365().get_dataframe_as_of(timestamp_at=snapshot_date)
    df = (
        df_out.join(df_map, df_out.PK_ACCOUNT == df_map.PK_ACCOUNT_CPQ, how="left")
        .drop(
            "FK_CLIENTE_SIEBEL",
            "FLG_CLI_MIGRATO",
            "FLG_CLI_D365",
            "STATO_CLI_D365",
            "DATA_MIGRAZIONE",
        )
        .withColumn(
            "COD_CLIENTE",
            F.when(F.col("PK_CLIENTE_D365").isNull(), F.col("PK_ACCOUNT")).otherwise(
                F.col("PK_CLIENTE_D365")
            ),
        )
        .drop("PK_CLIENTE_D365", "PK_ACCOUNT_CPQ")
    )
    out_cols = df_out.columns
    out_cols.insert(1, "COD_CLIENTE")
    return df.select(out_cols)


def get_customer_base(
    cb_path: str,
    snapshot_date: str,
    id_column: str,
    fornitura_column_in: str,
    fornitura_column_out: str,
) -> DataFrame:
    """Load and prepare customer base DataFrame."""
    cb = (
        spark.table(cb_path)
        .where(F.col("DATA_SNAPSHOT") == snapshot_date)
        .select("PK_ACCOUNT", id_column, fornitura_column_in)
        .distinct()
        .withColumnRenamed(fornitura_column_in, fornitura_column_out)
    )
    return cb


def get_inference_results(
    inference_results_conf: dict,
    snapshot_date: str,
    id_column: str,
    prediction_column: str,
    snapshot_date_col_in: str,
    snapshot_date_col_out: str,
    new_prediction_column: str,
) -> DataFrame:
    """Load and prepare inference results DataFrame."""
    table_path = f"{inference_results_conf['catalog']}.{inference_results_conf['schema']}.{inference_results_conf['table_name']}"
    df = (
        spark.table(table_path)
        .where(F.col("DATA_SNAPSHOT") == snapshot_date)
        .select(id_column, prediction_column, snapshot_date_col_in)
        .withColumnRenamed(snapshot_date_col_in, snapshot_date_col_out)
        .withColumnRenamed(prediction_column, new_prediction_column)
    )
    return df


def ensure_columns_and_cast(df: DataFrame, required_columns: dict) -> DataFrame:
    """Ensure all required columns exist and are cast to the correct type."""
    for col, d_type in required_columns.items():
        if col not in df.columns:
            logger.info("%s column is missing -> adding it with null values", col)
            df = df.withColumn(col, F.lit(None))
        logger.info("Casting %s column to %s dtype", col, d_type)
        df = df.withColumn(col, F.col(col).cast(d_type))
    return df
