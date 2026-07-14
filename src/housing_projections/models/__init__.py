from .models import ALL_MODELS

# Re-export every model class by name (M0, M0h, AZ0, ...) so existing
# `from housing_projections.models import M0` imports keep working.
# ALL_MODELS in models.py is the single source of truth for which models
# exist — add a model there and it appears here automatically.
globals().update(ALL_MODELS)

__all__ = ["ALL_MODELS", *ALL_MODELS]
