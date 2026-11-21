import dspy


class ExtractionSignature(dspy.Signature):
    """Estrai tutti i paragrafi dai seguenti due documenti in italiano. Restituisci una lista di liste, ogni lista contiene i 2 paragrafi corrispondenti ciascuno proveniente dai corrispettivi documenti. IGNORA le differenze di spazi multipli, tabulazioni e rientri. Se un paragrafo è presente solo in uno dei due documenti, inserisci il paragrafo disponibile e una stringa vuota ('') al posto del paragrafo mancante."""

    question: str = dspy.InputField()
    answer: list[list[str]] = dspy.OutputField(
        desc="Lista di liste, ogni lista contiene i 2 paragrafi corrispondenti dai due documenti in italiano."
    )


extractor = dspy.Predict(ExtractionSignature)
