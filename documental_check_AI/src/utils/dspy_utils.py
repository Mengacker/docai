import dspy


def configure_dspy_model(model_name):
    lm = dspy.LM(
        model=f"databricks/{model_name}",
        max_tokens=16384,
        temperature=0,
    )
    adapter = dspy.TwoStepAdapter(lm)
    return dspy.settings.configure(lm=lm, adapter=adapter)
