"""Utilities functions for saving stuff."""

import os
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame

from src.utils.logger import get_logger

logger = get_logger()

spark: SparkSession = SparkSession.builder.getOrCreate()


def save_uc_table(
    sdf: DataFrame,
    dbx_path: str,
    partition_column: str = None,
    table_format: str = "delta",
) -> None:
    """Save table in the unity catalog."""
    logger.info("Saving table on Unity Catalog")

    # Overwrite table - not partitioned
    if partition_column is None:
        sdf.write.format(table_format).mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(dbx_path)

    # Partitioned table
    else:
        # find partition current partition value
        partition_value = sdf.select(partition_column).distinct().collect()[0][0]
        sdf.write.format(table_format).partitionBy([partition_column]).mode(
            "overwrite"
        ).option(
            "replaceWhere", f"{partition_column}=='{partition_value}'"
        ).saveAsTable(dbx_path)

    logger.info("Dataset correctly saved in %s", dbx_path)
    return


def save_file_trigger(path: str) -> None:
    """Save an empty txt file to be used as trigger"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(path, f"log_{timestamp}.txt")
    # This creates an empty file
    with open(filepath, "w") as file:  # noqa:F841
        pass
