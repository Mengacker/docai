import re
import pyspark.sql.functions as F
from pyspark.sql import types as T


@F.udf(T.ArrayType(T.IntegerType()))
def check_identical_texts_udf(list_of_lists, use_regex=False):
    """
    Checks if the two texts in each sub-list are identical.

    Args:
        list_of_lists (list of lists): A list where each sub-list contains two strings to be compared.
        use_regex (bool): Whether to use regex for comparison. Default is False (uses direct equality).

    Returns:
        list: A list containing 0 if the two texts are identical and non-empty, and 1 otherwise.
              If either text is empty, it's considered non-identical.
    """
    results = []

    for pair in list_of_lists:
        if not pair or len(pair) != 2:
            results.append(1)
            continue

        text1, text2 = pair

        if not text1 or not text2:
            # Treat missing or empty strings as non-identical
            results.append(1)
        elif use_regex:
            try:
                results.append(0 if re.fullmatch(text1, text2) else 1)
            except re.error:
                results.append(1)
        else:
            results.append(0 if text1 == text2 else 1)

    return results
