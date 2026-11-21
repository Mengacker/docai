from mlflow.genai.scorers import Guidelines, scorer
import difflib

model_name = "check_documentale_gpt4o"


def validate_answer(example, pred, trace=None):
    scores = []
    # Handle nested list structure: list[list[str]]
    for pred_answer, example_answer in zip(pred.answer, example.answer):
        # Each pred_answer and example_answer is a list of strings
        if pred_answer == example_answer:
            scores.append(1)
        else:
            scores.append(0)
    return sum(scores) / len(scores)


def similarity(a: str, b: str) -> float:
    """Return a similarity score between 0 and 1."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def flatten_list(lst):
    """Flatten a list of lists into a single list."""
    if not lst:
        return []
    return [
        item
        for sublist in lst
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]


def validate_answer_V2(example: list, pred: list, trace=None) -> float:
    """
    Flexible comparison for Spark UDF (order-independent).
    - Flattens nested lists.
    - Normalizes text.
    - Greedily matches best prediction to each example.
    """
    threshold = 0.8
    score = 0.0

    pred_flat = [p for p in flatten_list(pred.answer)]
    example_flat = [e for e in flatten_list(example.answer)]

    if not example_flat:
        return -1.0  # No ground truth to compare against

    used_pred = set()  # keep track of matched predictions

    for ex in example_flat:
        best_sim = 0
        best_idx = -1
        for i, pr in enumerate(pred_flat):
            if i in used_pred:
                continue
            sim = similarity(ex, pr)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_sim >= threshold and best_idx != -1:
            score += 1
            used_pred.add(best_idx)
    result = float(score / len(example_flat))
    if trace is not None:
        return float(score / len(example_flat)) >= 1
    return result


@scorer
def validate_answer_V2_scorer(outputs: dict, expectations: dict) -> float:
    """
    Flexible comparison for Spark UDF (order-independent).
    - Flattens nested lists.
    - Normalizes text.
    - Greedily matches best prediction to each example.
    """
    threshold = 0.8
    score = 0.0
    output = outputs.get("risposta", {})
    pred_flat = [p for p in flatten_list(output)]
    example = expectations.get("ground_truth", {})
    example_flat = [e for e in flatten_list(example)]
    print(pred_flat)
    print(example_flat)

    if not example_flat:
        return -1.0  # No ground truth to compare against

    used_pred = set()  # keep track of matched predictions

    for ex in example_flat:
        best_sim = 0
        best_idx = -1
        for i, pr in enumerate(pred_flat):
            if i in used_pred:
                continue
            sim = similarity(ex, pr)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_sim >= threshold and best_idx != -1:
            score += 1
            used_pred.add(best_idx)
    result = float(score / len(example_flat))
    return result


scorers = [
    Guidelines(
        guidelines="La Response deve essere in Italiano",
        name="Italian_language",
        model=f"databricks:/{model_name}",
    ),
    Guidelines(
        guidelines="La Response deve contenere tutto il testo della Request anche se non nello stesso ordine, tranne le due parole 'CURRENT_DOCUMENT:' e 'PAST_DOCUMENT:'",
        name="Completeness",
        model=f"databricks:/{model_name}",
    ),
    Guidelines(
        guidelines="La Response deve essere una lista di liste, ogni lista contiene i 2 paragrafi corrispondenti dai due documenti",
        name="template_match",
        model=f"databricks:/{model_name}",
    ),
    validate_answer_V2_scorer,
]
