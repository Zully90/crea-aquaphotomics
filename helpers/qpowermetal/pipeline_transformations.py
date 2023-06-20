#
# QualiCtrl è un progetto di Qualitade S.r.l.
# (c) 2019 e anni successivi Qualitade S.r.l. - Ogni diritto è riservato.
# (c) 2019 and following years by Qualitade S.r.l. -  All right reserved.
#
# Autore: Marcello Vanzulli (marcello.vanzulli@qualitade.com)
# Autore: Nicholas Fiorentini (nicholas.fiorentini@qualitade.com)
#

"""
Helper functions to transform input on a scikit-learn's pipeline.

Use:

    from sklearn.pipeline import Pipeline
    from qpowermetal.pipeline_transformations import wavelenghts_cut_processing, derivate_processing

    model = Pipeline(steps=[
        ("wl_cut", wavelenghts_cut_processing(head=5, tail=10)),
        ("derivate", derivate_processing()),
    ])

    X_transformed = model.transform(X)
"""

from typing import Any, Union

import numpy as np
from numpy import ndarray
from scipy import signal
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import scale as skscale


# WL CUT
def _wl_cut(values: ndarray, head: int = 0, tail: int = 0) -> ndarray:
    """
    Removes head and tail columns.
    This function always returns a numpy array.

    values
        The data to be filtered as an array-like object (numpy). If x is not a single or double precision floating
        point array, it will be converted to type numpy.float64 before filtering.

    head
        Integer number of colums to remove from left to right. Es: head=2 removes the first two columns.
        Default: 0 (no removals)

    tail
        Integer number of colums to remove from right to left. Es: tails=2 removes the last two columns.
        Default: 0 (no removals)
    """
    # devo "reimpostare" tail per evitare che tagli tutti gli spettri se lascio il valore di default a 0
    if tail <= 0:
        tail = values.shape[1]
    else:
        tail = -tail

    return values[:, head:tail]


def wavelenghts_cut_processing(*, head: int = 5, tail: int = 5) -> FunctionTransformer:
    """Build the FunctionTranformer for the Pipeline."""
    return FunctionTransformer(
        _wl_cut, validate=True, kw_args={"head": head, "tail": tail,}
    )


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
    return signal.savgol_filter(
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


# SCALE and SNV
def scale(*, with_mean: bool = True, with_std: bool = True) -> FunctionTransformer:
    """Build the FunctionTranformer for the Pipeline."""
    return FunctionTransformer(
        skscale,
        validate=True,
        kw_args={"axis": 1, "with_mean": with_mean, "with_std": with_std},
    )


def snv_processing() -> FunctionTransformer:
    """Build the FunctionTranformer for the Pipeline."""
    return scale(with_mean=True, with_std=True)


# WL SELECTION
def _wl_selection(values: ndarray, indexes: ndarray) -> ndarray:
    """
    Select the wavelengths.

    values
        The data to be filtered as an array-like object (numpy). If x is not a single or double precision floating
        point array, it will be converted to type numpy.float64 before filtering.

    indexes
        List of indexes to select
    """
    return values[:, indexes]


def wavelength_selection_processing(
    *, selected_wl_indexes: ndarray
) -> FunctionTransformer:
    """Build the FunctionTranformer for the Pipeline."""
    return FunctionTransformer(
        _wl_selection, validate=True, kw_args={"indexes": selected_wl_indexes,}
    )


# NORMALIZZAZIONE
def _normalization(
    values: ndarray,
    centering: Union[ndarray, int, float] = 0,
    scaling: Union[ndarray, int, float] = 1,
) -> ndarray:
    """
    Normalize the values of the array applying centering and scaling. With centering = 0 and scaling = 1 the values are
    kept unchanged.

    centering: Number or array with shape coompatible with values
        Computes values - centering

    scaling: Number or array with shape coompatible with values
        Computes (values - centering) / scaling
        scaling must be not zero nor epsilon to avoid computational errors.
    """
    # better uniform the types
    if isinstance(scaling, (int, float,)):
        scaling = np.array([scaling,])

    # check scaling is not "too small": true iff at least one element of the array doesn't meet the condition
    if np.any([np.abs(scaling) <= np.finfo(float).eps]):
        raise ValueError("scaling cannot be 0 or too small.")

    return (values - centering) / scaling


def normalization(
    *, centering: Union[ndarray, int, float] = 0, scaling: Union[ndarray, int, float] = 1
) -> FunctionTransformer:
    """Build the FunctionTranformer for the Pipeline."""
    return FunctionTransformer(
        _normalization,
        validate=True,
        kw_args={"centering": centering, "scaling": scaling,},
    )


# MSC (Multiplicative Scatter Correction)
def _msc(values: ndarray, reference: ndarray) -> ndarray:
    """
    Computes and apply the Multiplicative Scatter Correction (MSC) to the input values.

    reference:
        Must be an array compatible with values for the linear regression.
    """

    def _internal(a: ndarray, *args: Any, **kwargs: Any) -> ndarray:
        """Internal helper to apply the computation on every item of the array."""
        try:
            # la polyfit estrae:  m * x + q
            m, q = np.polyfit(x=a, y=reference, deg=1)
            # m = 0 non dovrebbe mai accadere perché fallisce prima la polyfit (impossibile trovare la regressione)
        except np.linalg.LinAlgError as e:  # type: ignore[attr-defined]  # pragma: no cover
            # FIXME: per qualche oscuro motivo questo ramo non viene catturato dal coverage della pipeline, mentre
            #        funziona correttamente in locale. BOH!
            #        Notare anche che nella pipeline non ci sono gli output del tipo
            #           ** On entry to DLASCLS parameter number  4 had an illegal value
            raise ValueError(
                f"Linear regression not possible with the given values. {str(e)}"
            )

        return _normalization(values=a, centering=q, scaling=m)

    return np.apply_along_axis(func1d=_internal, axis=1, arr=values)


def msc(*, reference: ndarray) -> FunctionTransformer:
    """Build the FunctionTranformer for the Pipeline."""
    return FunctionTransformer(_msc, validate=True, kw_args={"reference": reference})
