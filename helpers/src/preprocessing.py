import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import FunctionTransformer
from numpy import ndarray

def _msc(data):
    
    try:
        # Convert DataFrame to numpy array
        spectra = data.to_numpy()
    except:
        spectra = data.copy()
        
    # Mean and std spectrum
    mean_spectrum = np.mean(spectra, axis=1, keepdims=True)
    std_spectrum = np.std(spectra, axis=1, keepdims=True)

    # Apply MSC correction
    msc_data = (spectra - mean_spectrum) / std_spectrum

    return msc_data


def msc_processing() -> FunctionTransformer:
    """Build the FunctionTranformer for the Pipeline."""
    return FunctionTransformer(_msc, validate=True)


# DERIVATE
def _derivate(
    values: ndarray, window_length: int = 7, polyorder: int = 2, derivate_order: int = 1
) -> ndarray:
    """
    Derivate of a spectral data using
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html

    This function always returns a numpy array.

    values
        The data to be filtered as an array-like object (numpy). If x is not a single or double precision floating
        point array, it will be converted to type numpy.float64 before filtering.

    window_length
        The length of the filter window (i.e., the number of coefficients). window_length must be a positive odd
        integer.
        Default: 7

    polyorder
        The order of the polynomial used to fit the samples. polyorder must be less than window_length.
        Default: 2

    derivate_order
        The order of the derivative to compute. This must be a nonnegative integer.
        Default: 1  (first degree derivate)
    """
    return savgol_filter(
        values, window_length=window_length, polyorder=polyorder, deriv=derivate_order
    )


def derivate_processing(
    *, window_length: int = 7, polyorder: int = 2, derivate_order: int = 1
) -> FunctionTransformer:
    """Build the FunctionTranformer for the Pipeline."""
    return FunctionTransformer(
        _derivate,
        validate=True,
        kw_args={
            "window_length": window_length,
            "polyorder": polyorder,
            "derivate_order": derivate_order,
        },
    )


# NORMALIZZAZIONE
def _norm(data):
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    standard_data = (pd.DataFrame(data) - mean ) / std
    
    return standard_data.values

def norm_preprocessing() -> FunctionTransformer:
    """Build the FunctionTranformer for the Pipeline."""
    return FunctionTransformer(_norm, validate=True)