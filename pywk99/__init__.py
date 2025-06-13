"""
Wavenumber-frequency analysis in Python.

Changelog
---------
Jun 13, 2025: Version 0.4.4
    - Allowing renaming of filter windows

Apr 8, 2025: Version 0.4.3
    - Adding support for healpix

Mar 7, 2025: Version 0.4.2
    - Organizing first github public version.

Feb 12, 2025: Version 0.4.1
    - Transferring repository to github.
    - Changing library name to pywk99

Dec 28, 2024: Version 0.4.0
    - Merging the functions filter_variable and
      filter_variable_multiple_windows together.
    - Removing type annotations of xr.Coordinates (not stable in xarray)
    - Fixing Fourier transform unit tests.
    - Adding library environmental file.
    - Updating library packaging mechanism to pyproject.toml
    - Adding notebook to download example data.
    - Modifying examples to make them compatible with the new version.
    - Adding functionality to plot filled filter windows.

Jul 02, 2024: Version 0.3.0
    - New core capability added to the package: Composite analysis of tropical
      waves in physical space
    - This justifies an upgrade of the minor version to 0.3.0

Mar 06, 2024: Version 0.2.0
    - Refactoring filter_variable_multiple_windows function so that it returns
      a xarray dataset instead of a list.
    - Changed to new minor version (0.2.0) as we changed the interface of an
      existing function.
    - Implemented an MJO window function using the same values in one of the
      example notebooks.
    - Added option to plot spectrum without taking the log.

Feb 26, 2024: Version 0.1.8
    - A new wave type for plotting and filtering: regular gravity waves.
    - Extension of the spectral analysis, i.e. Fourier transform and inverse
      Fourier transform, to 4-dimensional arrays including a 'height'
      dimension.
    - A new function which does the filtering of boxes or waves to separate a
      given wave's signal in physical space for multiple windows at once.
    - Adding the option to don't remove the seasonal cycle before filtering for
      a wave. Default behavior of the function doesn't change, however.
      Corresponding tests to these new functionalities.

Feb 16, 2024: Version 0.1.7
    - Refactoring of spectrum.py

Feb 13, 2024: Version 0.1.6
    - Merging bug fix for better segmentation of data.
    - Fixing spectrum tests.
    - Checking that coordinates are sorted before computing the power spectra
      or filtering a variable and raising a ValueError if not.

Sep 22, 2023: Version 0.1.4
    - Adding the possibility to make power spectrum analysis by season.
    - Making data_frequency an optional argument in get_power_spectrum.
    - Added new example to reproduce JJA power spectrum of Kiladis, Thorncroft,
      and Hall, 2006.
    - Organizing examples into own folder.

Sep 14, 2023: Version 0.1.0
    - First version with most important functionality covered by tests.
    - Several bugs where detected and corrected with the tests.
"""

__version__ = "0.4.4"
