from housing_projections.config import DEFAULT_SAMPLE_KWARGS

def run_model(model, results_dir='results/traces', **kwargs):
    """
    Convenience function: sample, print diagnostics, and save in one call.

    Parameters
    ----------
    model       : DwellingModel instance
    results_dir : str
    **kwargs    : override DEFAULT_SAMPLE_KWARGS

    Returns
    -------
    az.InferenceData
    """
    sample_kwargs = {**DEFAULT_SAMPLE_KWARGS, **kwargs}

    model.sample(**sample_kwargs)
    model.diagnostics()
    model.save(results_dir=results_dir)
    return model.trace