import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def msc(data):
    # Convert DataFrame to numpy array
    spectra = data.to_numpy()
    # Subtract mean spectrum from each spectrum in the data
    spectra -=  np.mean(spectra, axis=0)
    # Divide each spectrum in the data by the standard deviation spectrum
    spectra /= np.std(spectra, axis=0)
    # Convert back to DataFrame
    corrected_data = pd.DataFrame(spectra, columns=data.columns)

    return corrected_data


def smoothing(data, window_length=5, polyorder=2):
    smoothed_data = pd.DataFrame(columns=data.columns)
    for column in data.columns:
        smoothed_data[column] = savgol_filter(data[column], window_length, polyorder)
    return smoothed_data


def derivative(data, deriv_order=1):
    derivative_data = pd.DataFrame(columns=data.columns)
    for column in data.columns:
        derivative_spectrum = np.gradient(data[column], deriv_order)
        derivative_data[column] = derivative_spectrum
    return derivative_data


def normalization(data):
    # Column-wise standardization
    normalized_df=(df-df.mean())/df.std()
    return normalized_df