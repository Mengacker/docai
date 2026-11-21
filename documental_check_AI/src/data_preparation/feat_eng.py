"""Functions to create new features from existing ones"""

import pyspark.sql.functions as F


def clean_region_col(df, reg_col="CLI_REG"):
    """Clean the CLI_REG column"""
    df = df.withColumn(reg_col, F.lower(F.col(reg_col))).withColumn(
        reg_col,
        F.when(F.col(reg_col).startswith("emilia"), "emilia-romagna")
        .when(F.col(reg_col).startswith("friuli"), "friuli-venezia-giulia")
        .when(F.col(reg_col).startswith("trentino"), "trentino-alto-adige")
        .when(F.col(reg_col).startswith("valle"), "valle-aosta")
        .when(F.col(reg_col).isin("null", "it"), "missing")
        .otherwise(F.col(reg_col)),
    )
    return df


def clean_sdf_mod_ope(df, column_name):
    """Clean the column_name column"""
    df = df.withColumn(
        column_name,
        F.when(F.col(column_name).like("%SWITCH%"), "SWITCHIN")
        .when(F.col(column_name).like("%ATTIVAZION%"), "ATTIVAZIONE")
        .otherwise(F.col(column_name)),
    )
    return df


# not used
def clean_stato_sito_forn(df, column_name):
    """Clean the column_name column"""
    activated_cols = ["activated", "ativo", "attivo"]
    cessato_cols = [
        "cancelled",
        "cessato",
        "annullato",
        "non ativo",
        "non attivo",
        "suspended",
    ]
    other_cols = ["superseded", "potenziale"]
    pend_activation_cols = ["pending activation", "in attivazione"]
    df = df.withColumn(column_name, F.lower(F.col(column_name))).withColumn(
        column_name,
        F.when(F.col(column_name).isin(activated_cols), "attivo")
        .when(F.col(column_name).isin(cessato_cols), "cessato")
        .when(F.col(column_name).isin(other_cols), "altro")
        .when(F.col(column_name).isin(pend_activation_cols), "pending activation")
        .otherwise(F.col(column_name)),
    )
    return df


# not used
def clean_tipo_utilizzo(df, column_name):
    """Clean tipo_utilizzo to create a new column with the cleaned value."""
    df = (
        # Upper case for comparison
        df.withColumn(column_name, F.upper(F.col(column_name)))
        #  Start creating a new column with the cleaned value
        .withColumn(
            column_name,
            # subito tecnologico per intercettare tecnologico+riscaldamento
            F.when(F.col(column_name).like("%TECNOLOGICO%"), "tecnologico")
            # tripla acqua+cottura+riscaldamento
            .when(
                F.col(column_name).like("%ACQUA%")
                & F.col(column_name).like("%COTTURA%")
                & F.col(column_name).like("%RISCALDAMENTO%"),
                "acqua+cottura+riscaldamento",
            )
            # coppie
            .when(
                F.col(column_name).like("%ACQUA%")
                & F.col(column_name).like("%RISCALDAMENTO%")
                & ~F.col(column_name).like("%COTTURA%"),
                "acqua+riscaldamento",
            )
            .when(
                F.col(column_name).like("%ACQUA%")
                & ~F.col(column_name).like("%RISCALDAMENTO%")
                & F.col(column_name).like("%COTTURA%"),
                "acqua+cottura",
            )
            .when(
                ~F.col(column_name).like("%ACQUA%")
                & F.col(column_name).like("%RISCALDAMENTO%")
                & F.col(column_name).like("%COTTURA%"),
                "riscaldamento+cottura",
            )
            # singolo utilizzo
            .when(
                F.col(column_name).like("%ACQUA%")
                & ~F.col(column_name).like("%RISCALDAMENTO%")
                & ~F.col(column_name).like("%COTTURA%"),
                "acqua",
            )
            .when(
                ~F.col(column_name).like("%ACQUA%")
                & ~F.col(column_name).like("%RISCALDAMENTO%")
                & F.col(column_name).like("%COTTURA%"),
                "cottura",
            )
            .when(
                ~F.col(column_name).like("%ACQUA%")
                & F.col(column_name).like("%RISCALDAMENTO%")
                & ~F.col(column_name).like("%COTTURA%"),
                "riscaldamento",
            )
            .when(
                F.col(column_name).like("%FORNO%")
                | F.col(column_name).like("%CALDAIA%"),
                "forno/caldaia_individuale",
            )
            # condizionamento
            .when(F.col(column_name).like("%CONDIZION%"), "condizionamento")
            # se null lascia null -> ci pensa dopo la data preparation
            .when(F.col(column_name).isNull(), F.col(column_name))
            .otherwise("altro"),
        )
    )
    return df


# not used
def clean_forn_prec(df, column_name):
    """Clean fornitura precedente, for what is possible."""
    df = df.withColumn(column_name, F.lower(F.col(column_name))).withColumn(
        column_name,
        F.when(F.col(column_name).like("%enel%"), "enel")
        .when(
            F.col(column_name).like("%eni%") | F.col(column_name).like("%plenitude%"),
            "eni/plenitidue",
        )
        # se NON_DISPONIBILE count as null
        .when(F.col(column_name).like("%non disponibile%"), None)
        .when(F.col(column_name).like("%iren%"), "iren")
        .when(F.col(column_name).like("%edison%"), "edison")
        .when(F.col(column_name).like("%acea%"), "acea")
        .when(F.col(column_name).like("%hera%"), "hera")
        .when(F.col(column_name).like("%a2a%"), "a2a")
        .when(F.col(column_name).like("%e.on%"), "eon")
        .when(F.col(column_name).like("%illumia%"), "illumia")
        .when(F.col(column_name).like("%engie%"), "engie")
        .when(F.col(column_name).like("%bluenergy%"), "bluenergy")
        .when(F.col(column_name).like("%a2a%"), "a2a")
        .when(F.col(column_name).like("%optima%"), "optima")
        .when(F.col(column_name).like("%sorgenia%"), "sorgenia")
        .when(
            F.col(column_name).like("%servizio elettrico%")
            | F.col(column_name).like("%servizio energ%")
            | F.col(column_name).like("%compagnia energetic%"),
            "servizio_elettrico_nazionale",
        )
        .when(F.col(column_name).like("%green network%"), "green_network")
        .when(F.col(column_name).like("%poste%"), "poste")
        .when(F.col(column_name).like("%iberdrola%"), "iberdrola")
        .when(F.col(column_name).like("%dolomiti%"), "dolomiti")
        .when(F.col(column_name).like("%dufereco%"), "dufereco")
        .when(F.col(column_name).like("%octopus%"), "octopus")
        .when(F.col(column_name).isNull(), F.col(column_name))
        .otherwise("altri_fornitori"),
    )
    return df


def process_campaign_features(df):
    """Process campaign features to extract meaningful ones."""

    camcb_cols = [col for col in df.columns if col.startswith("CAMCB_CLI")]

    # Aggregate data
    df = (
        # Fill na first
        df.fillna(0, subset=camcb_cols)
        # TLS
        .withColumn(
            "CAMCB_CLI_NR_CAMP_TLS_RIF_9_MM",
            F.col("CAMCB_CLI_NR_CAMP_TLS_RIF_9_MM")
            + F.col("CAMCB_CLI_NR_CAMP_TLS_NUTI_9_MM"),
        )
        .drop("CAMCB_CLI_NR_CAMP_TLS_NUTI_9_MM")
        # DEM
        .withColumn(
            "CAMCB_CLI_NR_CAMP_DEM_RIF_9_MM",
            F.col("CAMCB_CLI_NR_CAMP_DEM_RIF_9_MM")
            + F.col("CAMCB_CLI_NR_CAMP_DEM_NUTI_9_MM"),
        )
        .drop("CAMCB_CLI_NR_CAMP_DEM_NUTI_9_MM")
        # SMS
        # .withColumn(
        #    "CAMCB_CLI_NR_SMS_TOT_9_MM",
        #    F.col("CAMCB_CLI_NR_CAMP_SMS_RIC_9_MM")
        #    + F.col("CAMCB_CLI_NR_CAMP_SMS_NRIC_9_MM"),
        # )
        # .drop(
        #    "CAMCB_CLI_NR_CAMP_SMS_RIC_9_MM",
        #    "CAMCB_CLI_NR_CAMP_SMS_NRIC_9_MM",
        # )
        # Inbound
        .withColumn(
            "CAMCB_CLI_NR_CAMP_INB_RIF_9_MM",
            F.col("CAMCB_CLI_NR_CAMP_INB_RIF_9_MM")
            + F.col("CAMCB_CLI_NR_CAMP_INB_NUTI_9_MM"),
        )
        .drop("CAMCB_CLI_NR_CAMP_INB_NUTI_9_MM")
        # Web
        # .withColumn(
        #    "CAMCB_CLI_NR_CAMP_WEB_TOT_9_MM",
        #    F.col("CAMCB_CLI_NR_CAMP_WEB_CLC_9_MM")
        #    + F.col("CAMCB_CLI_NR_CAMP_WEB_VIS_9_MM"),
        # )
        # .drop(
        #    "CAMCB_CLI_NR_CAMP_WEB_CLC_9_MM",
        #    "CAMCB_CLI_NR_CAMP_WEB_VIS_9_MM",
        # )
        # Rinnovi
        # .withColumnRenamed(
        #    "CAMCB_CLI_NR_CAMP_RIN_9_MM", "CAMCB_CLI_NR_CAMP_RIN_TOT_9_MM"
        # )
    )
    return df
