"""Functions used in the PostProcessor class."""

from pyspark.sql import functions as F
from pyspark.sql import types as T
import tiktoken
import re


@F.udf(T.StringType())
def remove_numbers_in_blocks(s: str) -> str:
    block_re = re.compile(r"Corrispettivi definiti dal venditore[\s\S]*?Vendita")
    # Numeri: 144, 0,1958, 0.1958, 1.234,56 (prima togliamo le cifre singole e con decimali)
    num_re = re.compile(r"\d+(?:[.,]\d+)?")
    if s is None:
        return None

    def _clean_block(m: re.Match) -> str:
        block = m.group(0)
        # rimuovi numeri nel blocco
        block = num_re.sub("", block)
        # (opzionale) normalizza spazi multipli e spazi prima di newline
        block = re.sub(r"[ \t]{2,}", " ", block)
        block = re.sub(r" *\n", "\n", block)
        return block

    return block_re.sub(_clean_block, s)


@F.udf(returnType=T.IntegerType())
def token_count_udf(string: str) -> int:
    """
    Returns the number of tokens in a text string.

    Args:
        text (str): The input text.

    Returns:
        int: The count of tokens in the text.
    """
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(string)
    return len(tokens)


@F.udf(returnType=T.StringType())
def remove_numbers(string: str) -> str:
    """
    Removes all numbers from a string.
    """
    return re.sub(r"\d+", "", string)


extract_and_clean_schema = T.StructType(
    [
        T.StructField("matches", T.ArrayType(T.StringType()), True),
        T.StructField("cleaned_text", T.StringType(), True),
    ]
)


@F.udf(returnType=extract_and_clean_schema)
def extract_and_clean_SR_udf(text):
    if not text:
        return {"matches": [], "cleaned_text": "no text"}

    # Updated patterns
    pattern = r"SCHEDA_RIEP_[A-Z0-9_]+_\d{8}_\d{8}"

    # Find matches (case-insensitive)
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    # Combine matches
    all_matches = set(matches)

    # Clean text (remove both patterns)
    cleaned = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Final cleanup of extra spaces
    cleaned = re.sub(r"\n{2,}", "", cleaned)

    return {"matches": all_matches, "cleaned_text": cleaned}


@F.udf(returnType=T.StringType())
def clean_text(text: str) -> str:
    if not text:
        return text

    stop_words = [
        "Condizioni Generali Di Contratto \n energia elettrica e/o gas naturale \n",
        "\n Condizioni Generali Di Contratto \n",
        "Condizioni Economiche Condizioni del mercato libero sottoscrivibili",
        "Condizioni Economiche Condizioni Economiche Condizioni sottoscrivibili",
        "Scheda Sintetica dell'offerta",
        "n° plico",
        "Codice Convenzione",
        "Codice incaricato",
    ]

    # --- Optimized date range removal pattern ---
    # Handles case-insensitive 'valida dal ... al ...'
    # Works with both normal and non-breaking spaces
    date_pattern = (
        r"(?i)\(?[\s\u00A0]*?(?:valida[\s\u00A0]+)?dal[\s\u00A0]*\d{2}/\d{2}/(?:\d{2}|\d{4})"
        r"[\s\u00A0]*al[\s\u00A0]*\d{2}/\d{2}/(?:\d{2}|\d{4})[\s\u00A0]*\)?"
    )

    cleaned = re.sub(date_pattern, "\n", text)

    # --- Stop words removal ---
    for word in stop_words:
        # Make stop-word pattern flexible to spaces and case
        normalized = re.sub(r"\s+", r"\\s*", word)
        pattern = rf"(?i){normalized}"  # case-insensitive
        cleaned = re.sub(pattern, " ", cleaned)

    return cleaned


def extract_tabella_valori(
    df, column_in: str, column_out: str, pattern: str, incident: str
):
    """
    Extracts tables/values from a text column based on a regex pattern,
    and removes those extracted tables from the original text.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame.
    column_in : str
        Name of the column containing text to search.
    column_out : str
        Name of the column where cleaned text (without tables) will be saved.
    pattern : str
        Regex pattern that identifies the table of values to extract.
        Should be a single pattern that matches the complete table.
    incident : str
        String separator used to split the text into separate "incidents"
        before applying the regex.

    Returns
    -------
    pyspark.sql.DataFrame
        A new DataFrame with:
        - "tabella_valori": array of all extracted matches from column_in
        - column_out: column_in text with all matches removed (incident structure preserved)
    """

    # Precompile regex once
    regex = re.compile(pattern, re.DOTALL)

    def extract_and_clean(text: str):
        if not text:
            return [], ""

        chunks = text.split(incident)
        results = []
        cleaned_chunks = []

        for chunk in chunks:
            found = regex.findall(chunk)
            results.extend(found)
            # remove matches from this chunk
            cleaned_chunk = regex.sub("", chunk)
            cleaned_chunks.append(cleaned_chunk)

        cleaned_text = incident.join(cleaned_chunks)
        return results, cleaned_text

    # Define UDF returning both array + cleaned string
    schema = T.StructType(
        [
            T.StructField("tabella_valori", T.ArrayType(T.StringType()), True),
            T.StructField("cleaned_text", T.StringType(), True),
        ]
    )

    extract_udf = F.udf(extract_and_clean, schema)

    # Apply UDF
    df = df.withColumn("extracted", extract_udf(F.col(column_in)))

    df = (
        df.withColumn("tabella_valori", F.col("extracted.tabella_valori"))
        .withColumn(column_out, F.col("extracted.cleaned_text"))
        .drop("extracted")
    )

    return df


@F.udf(T.BooleanType())
def filter_serving_rows_udf(text1, text2):
    if not text1 or not text2:  # catches None and empty strings
        return False

    # Remove specific prefixes
    text1 = text1.replace("CURRENT_DOCUMENT:", "").replace("PAST_DOCUMENT:", "")

    # Remove ALL occurrences of text2 (escaped for regex safety)
    remaining = re.sub(re.escape(text2), "", text1)

    # Remove all whitespace (spaces, tabs, newlines)
    remaining = re.sub(r"\s+", "", remaining)

    # Return True if remaining length > 2
    return len(remaining) > 2


def extract_two_tabelle_valori(
    df,
    column_in: str,
    column_out: str,
    pattern1: str,
    pattern2: str,
    incident: str,
):
    """
    Extracts two types of tables/values from a text column using two regex patterns,
    and removes them from the text, preserving incident structure.

    Returns a DataFrame with:
      - 'tabella_valori': list of tuples [(matches_type1, matches_type2), ...]
      - column_out: cleaned text with both table types removed
    """

    regex1 = re.compile(pattern1, re.MULTILINE)
    regex2 = re.compile(pattern2, re.MULTILINE)

    def extract_and_clean(text: str):
        if not text:
            return [], ""

        chunks = text.split(incident)
        results = []
        cleaned_chunks = []

        for chunk in chunks:
            found1 = regex1.findall(chunk)
            found2 = regex2.findall(chunk)

            # Collect as a tuple pair
            results.append((found1, found2))

            # Clean chunk from both matches
            cleaned_chunk = regex1.sub("", chunk)
            cleaned_chunk = regex2.sub("", cleaned_chunk)
            cleaned_chunks.append(cleaned_chunk)

        cleaned_text = incident.join(cleaned_chunks)
        return results, cleaned_text

    # Define return schema for UDF
    schema = T.StructType(
        [
            T.StructField(
                "tabella_valori",
                T.ArrayType(
                    T.StructType(
                        [
                            T.StructField(
                                "pattern1_matches", T.ArrayType(T.StringType()), True
                            ),
                            T.StructField(
                                "pattern2_matches", T.ArrayType(T.StringType()), True
                            ),
                        ]
                    )
                ),
                True,
            ),
            T.StructField("cleaned_text", T.StringType(), True),
        ]
    )

    # Helper to adapt Python tuple -> Spark struct
    def extract_and_clean_spark(text):
        pairs, cleaned = extract_and_clean(text)
        # Convert tuple to dict for Spark
        spark_pairs = [
            {"pattern1_matches": p1, "pattern2_matches": p2} for p1, p2 in pairs
        ]
        return spark_pairs, cleaned

    extract_udf = F.udf(extract_and_clean_spark, schema)

    df = df.withColumn("extracted", extract_udf(F.col(column_in)))

    df = (
        df.withColumn("tabella_valori", F.col("extracted.tabella_valori"))
        .withColumn(column_out, F.col("extracted.cleaned_text"))
        .drop("extracted")
    )

    return df
