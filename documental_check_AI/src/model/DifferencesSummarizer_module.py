import dspy


class DifferencesExtractionSignature(dspy.Signature):
    """
    Estrai le differenze di solo contenuto tra il CURRENT_DOCUMENT e il PAST_DOCUMENT.
    Ciascun elemento è una differenza del documento corrente rispetto a quello passato.
    Input: lista con 2 paragrafi, il primo paragrafo è il documento corrente e il secondo il documento passato.
    """

    paragraphs: str = dspy.InputField()
    list_of_difference: list[str] = dspy.OutputField(
        description="Lista di differenze, in italiano"
    )


DifferencesExtraction = dspy.Predict(DifferencesExtractionSignature)


class SummarySignature(dspy.Signature):
    """
    Riassumi solo le aggiunte e le assenze di parole e frazi citate nella lista.
    Non considerare la formattazione e la spaziattura.
    Se non viene rilevata alcuna differenza, rispondi con: "Nessuna differenza rilevata tra i due paragrafi."
    """

    list_of_difference: list[str] = dspy.InputField()
    history: dspy.History = dspy.InputField()
    summary: str = dspy.OutputField(description="Riassunto in italiano")


Summarizer = dspy.Predict(SummarySignature, temperature=0)


def summary(list_of_paragraphs: list[list[str]], comparison_lists: list[int]):
    results = []
    num_paragraph = []
    list_of_differences = []
    for i in range(len(comparison_lists)):
        if comparison_lists[i] == 1:
            differences = DifferencesExtraction(
                paragraphs=list_of_paragraphs[i]
            ).list_of_difference
            history = dspy.History(
                messages=[
                    {
                        "paragraphs": str(list_of_paragraphs[i]),
                        "list_of_difference": differences,
                    }
                ]
            )
            result = Summarizer(
                list_of_difference=list_of_paragraphs[i], history=history
            ).summary
            num_paragraph.append(i + 1)
            results.append(result)
            list_of_differences.append(differences)
        else:
            num_paragraph.append(i + 1)
            results.append("Nessuna differenza rilevata tra i due paragrafi.")
            list_of_differences.append(
                ["Nessuna differenza rilevata tra i due paragrafi."]
            )

    return [results, num_paragraph, list_of_differences]
