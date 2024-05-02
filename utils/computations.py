import numpy as np
import pywt
import torchaudio
from sklearn.impute import KNNImputer
from scipy.signal import butter, filtfilt

# Define a Spectrogram transform using torchaudio
spectrogram = torchaudio.transforms.Spectrogram(
    n_fft=800,        # Number of bins for FFT
    win_length=256,   # Window length for the spectrogram
    hop_length=44,    # Length of the hop between windows
    power=None        # Raw spectrogram output without any power operation
)

def mad(signal: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    """
    Calculate the Median Absolute Deviation (MAD), a robust measure of variability.

    Args:
        signal (np.ndarray): Input signal array.
        axis (int): The axis along which the median is computed. Default is -1 (last axis).
        keepdims (bool): Whether to keep the dimensions of the input array.

    Returns:
        np.ndarray: An array containing the MAD of the input signal.
    """
    median = np.median(signal, axis=axis, keepdims=keepdims)
    deviation = np.median(np.abs(signal - median), axis=axis, keepdims=keepdims)
    return deviation * 1.4826  # Constant to make the estimator consistent for normal distributions

def butter_filter(signal: np.ndarray, fs: int = 200, cutoff_freq: np.ndarray = np.array([0.25, 50]), order: int = 4, btype: str = "bandpass") -> np.ndarray:
    """
    Apply a Butterworth filter to a signal.

    Args:
        signal (np.ndarray): Input signal array.
        fs (int): Sampling frequency of the input signal.
        cutoff_freq (np.ndarray): Array of cutoff frequencies for the filter.
        order (int): Order of the filter.
        btype (str): Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop').

    Returns:
        np.ndarray: The filtered signal.
    """
    # Normalize the cutoff frequencies to the Nyquist frequency
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_freq / nyquist

    # Create the filter coefficients
    b, a = butter(N=order, Wn=normalized_cutoff, btype=btype, analog=False)
    # Apply the filter to the signal
    return filtfilt(b, a, signal)

def dwt_idwt(signal: np.ndarray, wavelet: str = 'db8') -> np.ndarray:
    """
    Apply Discrete Wavelet Transform (DWT) and Inverse Discrete Wavelet Transform (IDWT) for denoising.

    Args:
        signal (np.ndarray): Input signal.
        wavelet (str): Type of wavelet to use.

    Returns:
        np.ndarray: Signal reconstructed after DWT and IDWT.
    """
    coeffs = pywt.wavedec(signal, wavelet)
    coeffs[1:] = [np.zeros_like(coeff) for coeff in coeffs[1:]]  # Zeroing details for denoising
    reconstructed_signal = pywt.waverec(coeffs, wavelet)
    return reconstructed_signal

def knn_impute_spectrogram(spectrogram: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """
    Impute missing values in a spectrogram using K-Nearest Neighbors.
    
    Args:
        spectrogram (np.ndarray): The spectrogram with missing values (NaNs).
        n_neighbors (int): Number of neighbors to use for imputation.
    
    Returns:
        np.ndarray: Spectrogram with imputed values.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    spectrogram_imputed = imputer.fit_transform(spectrogram.reshape(-1, spectrogram.shape[-1])).reshape(spectrogram.shape)
    return spectrogram_imputed

def bin_array(array: np.ndarray, bin_size: int, axis: int = -1, pad_dir: str = "symmetric", mode: str = "edge", **kwargs) -> np.ndarray:
    """
    Bin an array along a specified axis into equal-sized bins.

    Args:
        array (np.ndarray): Input array to bin.
        bin_size (int): Number of elements in each bin.
        axis (int): Axis along which to bin the array.
        pad_dir (str): Direction to pad the array ('left', 'right', 'symmetric').
        mode (str): Padding mode as per numpy's pad function.

    Returns:
        np.ndarray: Array reshaped into bins with possible padding applied.
    """
    if axis == -1:
        axis = array.ndim - 1

    current_length = array.shape[axis]
    num_bins = math.ceil(current_length / bin_size)
    new_length = num_bins * bin_size

    # Calculate padding if the current length doesn't divide evenly into bins
    padding_amount = new_length - current_length
    pad_left = pad_right = 0

    if pad_dir == "left":
        pad_left = padding_amount
    elif pad_dir == "right":
        pad_right = padding_amount
    else:  # symmetric
        pad_left = padding_amount // 2
        pad_right = padding_amount - pad_left

    # Apply padding to the array
    padding_config = [(0, 0)] * array.ndim
    padding_config[axis] = (pad_left, pad_right)
    array_padded = np.pad(array, pad_width=padding_config, mode=mode, **kwargs)

    # Reshape the array to new dimensions with bins
    new_shape = list(array.shape)
    new_shape[axis] = num_bins
    new_shape.insert(axis + 1, bin_size)
    array_binned = array_padded.reshape(new_shape)

    return array_binned
