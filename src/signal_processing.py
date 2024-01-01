import librosa
import numpy as np


def load_q_transform_audio_file(
    audio_filepath: str,
    start_time_s: float,
    duration_s: float,
    sample_rate: int = 48000,
):
    """
    Load an audio file and perform a Constant-Q Transform (CQT) on it.

    This function loads an audio file from a specified path, starting at a defined start time and for a given duration.
    It then performs a CQT on the loaded audio signal. The CQT is adjusted in magnitude and converted to a decibel scale.
    Finally, the CQT is limited in range for further processing.

    Parameters:
    - audio_filepath (str): The file path to the audio file.
    - start_time_s (float): The start time in seconds from which the audio should be loaded.
    - duration_s (float): The duration in seconds for which the audio should be loaded.
    - sample_rate (int, optional): The sample rate to be used for the CQT. Defaults to 48,000 Hz.

    Returns:
    - new_CQT (np.ndarray): The Constant-Q Transform of the audio file, processed and ready for further analysis.

    The CQT is computed using the librosa library. The magnitude of the CQT is raised to the fourth power and
    converted to a decibel scale. The CQT is then limited in range using the `cqt_lim` function, which is not
    defined within this function and should be provided externally.

    Note: This function assumes that `librosa` and `numpy` are imported as `librosa` and `np`, respectively.
    The `cqt_lim` function must also be defined or imported in the context where this function is used.
    """
    # Perform the Constant-Q Transform
    data, sr = librosa.load(
        audio_filepath, sr=None, mono=True, offset=start_time_s, duration=duration_s
    )
    CQT = librosa.cqt(
        data, sr=sample_rate, hop_length=1024, fmin=None, n_bins=96, bins_per_octave=12
    )
    CQT_mag = librosa.magphase(CQT)[0] ** 4
    CQTdB = librosa.core.amplitude_to_db(CQT_mag, ref=np.amax)
    return CQTdB
