"""Compute theoretical dispersion relations of dry equatorial waves."""

from math import sqrt
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

EARTH_RADIUS = 6371000.0                    # earth radius [m]
DAY = 86400.0                               # day length [s]
OMEGA = 2.0 * np.pi / DAY                   # earth angular velocity [rad/s]
BETA_EQUATOR = 2.0 * OMEGA / EARTH_RADIUS   # beta parameter at eq [rad/(s*m)]
GRAVITY = 9.8067                            # gravitational acceleration [m/s2]
SYMMETRIC_WAVE_TYPES = ["inertio_gravity", "kelvin", "equatorial_rossby"]
ASYMMETRIC_WAVE_TYPES = ["inertio_gravity", "mixed_rossby_gravity"]
GENERAL_WAVE_TYPES = ["gravity"]
WAVE_TYPES = list(set(SYMMETRIC_WAVE_TYPES).union(set(ASYMMETRIC_WAVE_TYPES)).union(
    set(GENERAL_WAVE_TYPES)))


@dataclass
class LinearWave:
    """Wave of a given type, equivalent depth and Hermite polynomial number."""
    wave_type: str
    equivalent_depth: float
    n_polynomial: Optional[int] = 1

    def frequency(self, wave_number: np.ndarray) -> np.ndarray:
        """Get the wave frequency in CPD of provided zonal wave numbers."""
        nondim_wave_number = wave_number*self.lenght_scale/EARTH_RADIUS
        nondim_omega = _nondim_dispersion_relation(
            self.wave_type,
            nondim_wave_number,
            n_polynomial=self.n_polynomial
        )
        omega = nondim_omega/(self.time_scale) # rad/s
        omega = DAY*omega/(2*np.pi)            # rad/s*86400s/1d*2pi/rad = 1/d
        return omega

    def __post_init__(self) -> None:
        if not self.wave_type in WAVE_TYPES:
            raise ValueError(f"wave type must be one of {WAVE_TYPES}")

    @property
    def gravity_wave_speed(self) -> float:
        gravity_wave_speed = sqrt(GRAVITY*self.equivalent_depth)
        return gravity_wave_speed

    @property
    def time_scale(self) -> float:
        time_scale = sqrt(1/(BETA_EQUATOR*self.gravity_wave_speed))
        return time_scale

    @property
    def lenght_scale(self) -> float:
        lenght_scale = sqrt(self.gravity_wave_speed/BETA_EQUATOR)
        return lenght_scale


def wk_waves(component_type: str,
             equivalent_depths: list[float]):
    """Get the symmetric or asymmetric wave types used in wk99"""
    wave_names, n_polynomial = _wk_curves_names(component_type)
    waves = []
    for equivalent_depth in equivalent_depths:
        for wave_name in wave_names:
            wave = LinearWave(wave_name, equivalent_depth, n_polynomial)
            waves.append(wave)
    return waves

def individual_waves(wave_list: list[Tuple[str, int]],
                     equivalent_depths: list[float]):
    """Get the user-specified wave types"""
    waves = []
    for equivalent_depth in equivalent_depths:
        for wave_tuple in wave_list:
            wave_name = wave_tuple[0]
            n_polynomial = wave_tuple[1]
            wave = LinearWave(wave_name, equivalent_depth, n_polynomial)
            waves.append(wave)
    return waves


def _wk_curves_names(component_type: str) -> Tuple[list[str], int]:
    """Get curves and polynomial number as used in WK99 figures."""
    if component_type == "symmetric":
        waves = SYMMETRIC_WAVE_TYPES
        n_polynomial = 1
    elif component_type == "asymmetric":
        waves = ASYMMETRIC_WAVE_TYPES
        n_polynomial = 2
    else:
        raise ValueError(f"Unknown component_type '{component_type}'")
    return waves, n_polynomial


def _nondim_dispersion_relation(wave_type: str,
                                wave_number: np.ndarray,
                                n_polynomial: Optional[int] = 1) -> np.ndarray:
    """Get the non-dimensional frequency corresponding to a wave number"""
    dispersion_relations = {
        "gravity": _nondim_gravity_dispersion,
        "kelvin": _nondim_kelvin_dispersion,
        "inertio_gravity": _nondim_inertio_gravity_dispersion,
        "equatorial_rossby": _nondim_equatorial_rossby_dispersion,
        "mixed_rossby_gravity": _nondim_mixed_rossby_gravity_dispersion}
    dispersion_relation = dispersion_relations[wave_type]
    omega = dispersion_relation(wave_number, n_polynomial=n_polynomial)
    return omega


def _nondim_gravity_dispersion(wave_number: np.ndarray, **kwargs) -> np.ndarray:
    """Non dimensional dispersion relation of regular Gravity Waves."""
    omega = abs(wave_number)
    return omega


def _nondim_kelvin_dispersion(wave_number: np.ndarray, **kwargs) -> np.ndarray:
    """Non dimensional dispersion relation of Kelvin Waves."""
    omega = wave_number.copy()
    omega[omega < 0] = np.nan
    return omega


def _nondim_inertio_gravity_dispersion(wave_number: np.ndarray,
                                       n_polynomial: int = 1) -> np.ndarray:
    """Non dimensional dispersion relation of Inierto-Gravity Waves."""
    omega = np.sqrt(wave_number**2 + 2*n_polynomial + 1)
    return omega


def _nondim_equatorial_rossby_dispersion(wave_number: np.ndarray,
                                         n_polynomial: int = 1) -> np.ndarray:
    """Non dimensional dispersion relation of Equatorial Rossby Waves."""
    denominator = wave_number**2 + 2*n_polynomial + 1
    omega = -wave_number/denominator
    omega[omega < 0] = np.nan
    return omega


def _nondim_mixed_rossby_gravity_dispersion(wave_number: np.ndarray,
                                            **kwargs) -> np.ndarray:
    """Non dimensional dispersion relation of Mixed Rossby Gravity Waves."""
    term1 = wave_number/2
    term2 = np.sqrt((wave_number/2)**2 + 1)
    omega = term1 + term2
    return omega
