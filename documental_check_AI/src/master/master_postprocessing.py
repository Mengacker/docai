"""Functions used in the PostProcessor class."""

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from typing import List

from src.data_preparation.utils import cast_type_and_trim_strings
from src.data_preparation.feat_eng import (
    clean_region_col,
    clean_sdf_mod_ope,
    process_campaign_features,
)
from src.utils.utils import identify_variable_types


def process_metriche_cliente(
    spine: DataFrame, first_master: DataFrame, metriche_cols: List[str], id_col: str
) -> DataFrame:
    """
    Aggregate metrics with account keys, keeping only user-defined cols.
    """
    df_clien = first_master.select(id_col, *metriche_cols).dropDuplicates([id_col])
    disc, cat, cont, other = identify_variable_types(df_clien, [id_col])
    numerical_fields = disc + cont + [m for m in other if m != id_col]

    # Cast and trim, clean region field, process campaign features
    df_clien = cast_type_and_trim_strings(df_clien, [id_col], numerical_fields, cat)
    df_clien = clean_region_col(df_clien)
    df_clien = process_campaign_features(df_clien)

    # Encode variables like CLI_FLG_REG_PORT and FLG_MACTO_MICROBUSINESS
    df_clien = df_clien.withColumn(
        "CLI_FLG_REG_PORT", F.when(F.col("CLI_FLG_REG_PORT") == "Y", 1).otherwise(0)
    )
    if "CLI_MACTO" in df_clien.columns:
        df_clien = df_clien.withColumn(
            "FLG_MACTO_MICROBUSINESS",
            F.when(F.col("CLI_MACTO") == "MICROBUSINESS", 1).otherwise(0),
        ).drop("CLI_MACTO")

    # Fillna for specific fields where nan equals 0
    rdsat_cols = [c for c in df_clien.columns if c.startswith("RDSAT")]
    promo_cols = [c for c in numerical_fields if c.startswith("OFORD_CLI")]
    ga4_cols = [
        c for c in metriche_cols if c.startswith("DIGIT_CLI") or c.startswith("TOTAL_")
    ]
    if rdsat_cols:
        df_clien = df_clien.fillna(0, rdsat_cols)
    if "CLI_FLG_OPZ_VERDE" in df_clien.columns:
        df_clien = df_clien.fillna(0, ["CLI_FLG_OPZ_VERDE"])
    if promo_cols:
        df_clien = df_clien.fillna(0, promo_cols)
    servizi_cols = [c for c in numerical_fields if c.startswith("NR_SERVIZI")]
    if servizi_cols:
        df_clien = df_clien.fillna(0, servizi_cols)
    if ga4_cols:
        df_clien = df_clien.fillna(0, ga4_cols)

    return spine.join(df_clien, on=id_col, how="left")


def process_metriche_pdf(
    spine: DataFrame, first_master: DataFrame, metriche_cols: List[str], id_col: str
) -> DataFrame:
    """
    Process metriche with PDF key.
    """
    df = (
        first_master.select(id_col, *metriche_cols)
        .groupby(id_col)
        .agg(F.max("NR_RINNOVI_PDF").alias("NR_RINNOVI_PDF"))
    )
    return spine.join(df, on=id_col, how="left").fillna(0, subset=["NR_RINNOVI_PDF"])


def process_metriche_sdf(
    spine: DataFrame, first_master: DataFrame, metriche_cols: List[str], id_col: str
) -> DataFrame:
    """
    Process sito di fornitura metriche.
    """
    df_sdf = first_master.select(id_col, *metriche_cols).dropDuplicates([id_col])
    disc, cat, cont, other = identify_variable_types(df_sdf, [id_col])
    numerical_fields = disc + cont

    df_sdf = cast_type_and_trim_strings(df_sdf, [id_col], numerical_fields, cat)
    df_sdf = df_sdf.fillna(0, numerical_fields)
    df_sdf = clean_sdf_mod_ope(df_sdf, "SDF_MOD_OPE")

    return spine.join(df_sdf, on=id_col, how="left")


def process_metriche_conto_fattura(
    spine: DataFrame, first_master: DataFrame, metriche_cols: List[str], id_col: str
) -> DataFrame:
    """
    Process metriche conto fattura.

    Un conto fattura può contenere più PDF quindi non c'è bisogno di aggregare.

    WebBolletta, BillingDS, Payment, MetricheDS, ContoClienteInteraction.
    """
    df_conto_fatt = first_master.select(id_col, *metriche_cols).dropDuplicates([id_col])
    disc, cat, cont, other = identify_variable_types(df_conto_fatt, [id_col])
    numerical_fields = disc + cont

    df_conto_fatt = cast_type_and_trim_strings(
        df_conto_fatt, [id_col], numerical_fields, cat
    )
    df_conto_fatt = df_conto_fatt.fillna(0, numerical_fields)

    # Clean CC_FREQ_PAY_CURR
    if "CC_FREQ_PAY_CURR" in df_conto_fatt.columns:
        df_conto_fatt = df_conto_fatt.withColumn(
            "CC_FREQ_PAY_CURR",
            F.when(F.col("CC_FREQ_PAY_CURR") == "ND", None).otherwise(
                F.col("CC_FREQ_PAY_CURR")
            ),
        )
    return spine.join(df_conto_fatt, on=id_col, how="left")


def process_metriche_ali(
    spine: DataFrame, first_master: DataFrame, metriche_cols: List[str], id_col: str
) -> DataFrame:
    """
    Aggregates metrica with key asset_line_item.
    """
    df = (
        first_master.select(id_col, *metriche_cols)
        .dropDuplicates([id_col])
        .fillna(0, subset=["FORN_FLG_EX_REGTO"])
    )
    return spine.join(df, on=id_col, how="left")


def process_metriche_pod_pdr(
    spine: DataFrame, first_master: DataFrame, metriche_cols: List[str], id_col: str
) -> DataFrame:
    """
    Aggregates metrica with key pod_pdr.
    """
    df = (
        first_master.select(id_col, *metriche_cols)
        .groupby(id_col)
        .agg(F.max("NR_SWOUT_PRECEDENTI").alias("NR_SWOUT_PRECEDENTI"))
    )
    return spine.join(df, on=id_col, how="left").fillna(
        0, subset=["NR_SWOUT_PRECEDENTI"]
    )


def process_prodotto_attributi(
    spine: DataFrame, first_master: DataFrame, metriche_cols: List[str], id_col: str
) -> DataFrame:
    """
    Aggregates metrica prodotto attributi.
    """
    df = first_master.select(id_col, *metriche_cols)
    # F.mode is not a standard PySpark function; if using Spark 3.4+, use F.mode, else fallback to F.first or custom logic.
    df = df.groupby(id_col).agg(
        F.mode("PROD_TIP_OFF").alias("PROD_TIP_OFF"),
        F.mode("PROD_FAMIGLIA").alias("PROD_FAMIGLIA"),
        # F.mode("PROD_MERC_FINE_TUTELA").alias("PROD_MERC_FINE_TUTELA"),
    )
    return spine.join(df, on=id_col, how="left")
