import numpy as np
import pandas as pd
import libpysal
from esda.moran import Moran

from libpysal.weights import Queen
from sklearn.base import BaseEstimator, TransformerMixin


# ── Weights matrix ────────────────────────────────────────────────────────────


def weights_to_dense(w):
    """Convert libpysal weights to dense numpy array."""
    return libpysal.weights.full(w)[0]


def build_weights_libpysal(gdf):
    """
    Build row-normalised Queen contiguity weights as a libpysal object.
    Use this when you need a libpysal weights object (e.g. for Moran's I).
    """
    w           = Queen.from_dataframe(gdf, silence_warnings=True, use_index=False)
    w.transform = 'r'
    return w


def build_spatial_weights(gdf):
    """
    Build row-normalised queen contiguity spatial weights matrix.
    Returns dense numpy array of shape (n_areas, n_areas).
    Use this when you need a numpy matrix (e.g. for PyMC/pytensor).
    """
    w = build_weights_libpysal(gdf)
    W = np.zeros((len(gdf), len(gdf)))
    for i, neighbours in w.neighbors.items():
        for j, wij in zip(neighbours, w.weights[i]):
            W[i, j] = wij
    return W


# ── Spatial lag features ──────────────────────────────────────────────────────

def add_spatial_lag_features(gdf, feature_cols, use_index=False):
    """
    Compute Queen contiguity spatial lags for the given feature columns
    and append them to the GeoDataFrame with 'lag_{col}' names.

    Parameters
    ----------
    gdf          : GeoDataFrame
    feature_cols : list of str
    use_index    : bool

    Returns
    -------
    GeoDataFrame with additional lag columns
    """
    w       = build_weights_libpysal(gdf)
    W_dense = weights_to_dense(w)

    gdf_out = gdf.copy()
    for col in feature_cols:
        gdf_out[f'lag_{col}'] = W_dense @ gdf[col].values

    return gdf_out


# ── Sklearn transformer ───────────────────────────────────────────────────────

class SpatialLagTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that computes spatial lag features
    from a GeoDataFrame. Expects X to be a GeoDataFrame — computes
    a Queen contiguity weights matrix from the geometries and appends
    spatially lagged versions of the specified feature columns.

    Parameters
    ----------
    feature_cols : list of str — columns to use as features
    lag_cols     : list of str or None — columns to lag (default: all feature_cols)
    """

    def __init__(self, feature_cols, lag_cols=None):
        self.feature_cols = feature_cols
        self.lag_cols     = lag_cols

    def fit(self, X, y=None):
        self.lag_cols_ = self.lag_cols if self.lag_cols is not None \
                         else self.feature_cols
        return self

    def transform(self, X):
        w       = build_weights_libpysal(X)
        W_dense = weights_to_dense(w)

        feature_arr = X[self.feature_cols].values
        lag_arr     = W_dense @ X[self.lag_cols_].values

        return np.column_stack([feature_arr, lag_arr])


# ── Moran's I ─────────────────────────────────────────────────────────────────

def morans_i(values, w, permutations=999):
    """
    Compute Moran's I for a vector of values given a weights matrix.

    Parameters
    ----------
    values       : array-like (n_areas,)
    w            : libpysal weights object
    permutations : int

    Returns
    -------
    dict with keys 'I', 'p_value', 'z_score'
    """

    m = Moran(values, w, permutations=permutations)
    return {
        'I':       m.I,
        'p_value': m.p_sim,
        'z_score': m.z_sim,
    }


def morans_i_by_year(values_by_year, w, permutations=999):
    """
    Compute Moran's I for each year independently.

    Parameters
    ----------
    values_by_year : (n_areas, n_years)
    w              : libpysal weights object

    Returns
    -------
    pd.DataFrame with columns I, p_value, z_score indexed by year
    """
    results = [
        morans_i(values_by_year[:, t], w, permutations=permutations)
        for t in range(values_by_year.shape[1])
    ]
    return pd.DataFrame(results)