"""Functions utilities to prepare data for the model"""

from pyspark.sql import SparkSession

from src.data_preparation.utils import (
    split_train_test,
    cast_column_type,
    get_cols_to_drop_based_on_nan,
)
from src.utils.logger import get_logger
from src.utils.save_utils import save_uc_table

spark: SparkSession = SparkSession.builder.getOrCreate()
logger = get_logger()


def prepare_dataset_for_training(
    df,
    features_list,
    category_cols,
    target_col,
    output_table_name,
    params,
    save_table=True,
):
    """Prepare the dataset for training by splitting X, y and setting datatypes."""

    df = df.select(*features_list, target_col)

    # 1. Drop duplicates
    logger.info("Dataset count raw: %s", df.count())
    df = df.dropDuplicates()
    df.cache().count()  # cache for speeding up
    logger.info("Dataset count after dropping duplicates: %s", df.count())

    # 2. Drop columns with too much nulls (based on cutoff)
    cols_to_drop = get_cols_to_drop_based_on_nan(
        df, features_list, threshold_ratio=params["cutoff_nulls"]
    )
    features_list = [c for c in features_list if c not in cols_to_drop]
    logger.info("Following columns will be dropped: %s", cols_to_drop)
    df = df.drop(*cols_to_drop)

    # 3. Split in train - test
    df = split_train_test(df, test_ratio=0.2, target_col=target_col, seed=42)

    # 4. Cast data types
    df = cast_column_type(df, category_cols)

    # Save table
    if save_table:
        save_uc_table(df, dbx_path=output_table_name)
    return df


def prepare_dataset_for_inference(df, category_cols):
    """Apply the same processing steps to inference dataset."""
    df = cast_column_type(df, category_cols)
    return df
