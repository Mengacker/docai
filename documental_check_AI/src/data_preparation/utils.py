"""Utility functions to help prepare data for the model."""

import pyspark.sql.functions as F
from pyspark.sql import types as T
from pyspark.sql import DataFrame

from src.utils.logger import get_logger
from src.utils.utils import get_target_percentage

logger = get_logger()


def split_train_test(
    dataset: DataFrame, test_ratio: float, target_col: str, seed: int = 42
) -> DataFrame:
    """
    Split the dataset, creating a column identifier for test and train sets.

    Args:
        dataset: Input Spark DataFrame.
        test_ratio: Proportion of each class to include in the test set.
        target_col: Name of the target column.
        seed: Random seed.

    Returns:
        DataFrame with 'is_test_set' column (0 for train, 1 for test).
    """
    test_df = dataset.sampleBy(
        target_col, fractions={0: test_ratio, 1: test_ratio}, seed=seed
    )
    train_df = dataset.subtract(test_df).withColumn("is_test_set", F.lit(0))
    logger.info(
        "Raw dataset target percentage: %s",
        get_target_percentage(dataset, [target_col])[target_col],
    )
    logger.info(
        "Test target percentage: %s",
        get_target_percentage(test_df, [target_col])[target_col],
    )
    return train_df.union(test_df.withColumn("is_test_set", F.lit(1)))


def get_cols_to_drop_based_on_nan(
    df: DataFrame, columns: list, threshold_ratio: float = 0.5
) -> list:
    """
    Identify columns to drop based on the proportion of NaN/null values.

    Args:
        df: Input Spark DataFrame.
        columns: List of columns to check.
        threshold_ratio: Drop columns with more than this ratio of nulls.

    Returns:
        List of column names to drop.
    """
    logger.info("Cutoff threshold: %s", threshold_ratio)
    threshold = threshold_ratio * df.count()
    cols_to_drop = [
        col for col in columns if df.filter(df[col].isNull()).count() > threshold
    ]
    return cols_to_drop


def get_categorical_features(metriche_conf: dict, df: DataFrame) -> list:
    """
    Extract categorical features present in df from metriche_conf.

    Args:
        metriche_conf: Configuration dictionary with feature types.
        df: Input Spark DataFrame.

    Returns:
        List of categorical feature names present in df.
    """
    categorical_feats = [
        field
        for v in metriche_conf.values()
        for field, dtype in v.items()
        if dtype == "categorical"
    ]
    return [c for c in categorical_feats if c in df.columns]


def cast_column_type(df: DataFrame, categorical_features: list) -> DataFrame:
    """
    Cast datatypes to reduce memory and boost algorithm performance.

    Args:
        df: Input Spark DataFrame.
        categorical_features: List of categorical feature names.

    Returns:
        DataFrame with casted columns.
    """
    for column, dtype in df.dtypes:
        if column in categorical_features:
            df = df.withColumn(column, F.col(column).cast(T.StringType()))
        elif "FLG" in column or "NR" in column:
            df = df.withColumn(column, F.col(column).cast(T.IntegerType()))
        elif dtype == "bigint":
            df = df.withColumn(column, F.col(column).cast(T.IntegerType()))
        elif dtype == "double" or dtype == "decimal":
            df = df.withColumn(column, F.col(column).cast(T.FloatType()))
    return df


def cast_type_and_trim_strings(
    df: DataFrame, id_cols: list, numerical_fields: list, categorical_fields: list
) -> DataFrame:
    """
    Cast the correct data type and trim string columns.

    Args:
        df: Input Spark DataFrame.
        id_cols: List of identifier columns.
        numerical_fields: List of numerical field names.
        categorical_fields: List of categorical field names.

    Returns:
        DataFrame with casted and trimmed columns.
    """
    df = df.select(
        *id_cols,
        *(F.col(g).cast(T.LongType()).alias(g) for g in numerical_fields),
        *(F.col(g).cast(T.StringType()).alias(g) for g in categorical_fields),
    )

    df = df.select(
        *id_cols,
        *(F.trim(F.col(g)).alias(g) for g in categorical_fields),
        *(numerical_fields),
    )
    return df
