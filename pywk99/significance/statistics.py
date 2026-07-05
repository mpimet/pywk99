import numpy as np

from pywk99.spectrum.background import smooth_spectrum
from pywk99.spectrum.spectrum import _coherence_squared


def log_distance_statistic(spectra_a, spectra_b):
    wk_spectrum_a = sum(spectra_a) / len(spectra_a)
    wk_spectrum_b = sum(spectra_b) / len(spectra_b)
    log_distance = np.log10(wk_spectrum_b / wk_spectrum_a)
    return log_distance


def coh2_distance_statistic(crs_spectra_a, crs_spectra_b):
    smoothing_passes = 10
    coh2_a = sum(crs_spectra_a) / len(crs_spectra_a)
    coh2_b = sum(crs_spectra_b) / len(crs_spectra_b)
    coh2_a = _coherence_squared(coh2_a)
    coh2_b = _coherence_squared(coh2_b)
    coh2_a = smooth_spectrum(coh2_a, smoothing_passes)
    coh2_b = smooth_spectrum(coh2_b, smoothing_passes)
    distance = coh2_b - coh2_a
    return distance
