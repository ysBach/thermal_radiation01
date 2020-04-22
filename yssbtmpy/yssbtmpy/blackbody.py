import numpy as np
from astropy import units as u
from .constants import CC, HH, KB, SIGMA_SB
from .util import change_to_quantity


__all__ = ["B_lambda", "b_lambda"]


def B_lambda(wavelen, temperature):
    ''' Calculates black body radiance [energy/time/area/wavelen/sr]
    Parameters
    ----------
    wavelen : float or ~Quantity, ~numpy.ndarray of such.
        The wavelengths. In meter unit if not Quantity.
    temperature : float
        The temperature. In Kelvin unit if not Quantity. For specific
        purpose, you can give it in an ndarray format, but not
        recommended.

    Return
    ------
    radiance : ndarray
        The black body radiance [energy/time/area/wavelen/sr].
    '''
    wl = change_to_quantity(wavelen, u.m, to_value=True)
    temp = change_to_quantity(temperature, u.K, to_value=True)
    coeff = 2*HH*CC**2 / wl**5
    denom = np.exp(HH*CC/(wl*KB*temp)) - 1
    radiance = coeff / denom
    return radiance


def b_lambda(wavelen, temperature):
    ''' Calcualtes the small b function [1/wavelen].
    Parameters
    ----------
    wavelen : float or ~Quantity, ~numpy.ndarray of such.
        The wavelengths. In meter unit if not Quantity.
    temperature : float or ~Quantity
        The temperature. In Kelvin unit if not Quantity. For specific
        purpose, you can give it in an ndarray format, but not
        recommended.

    Return
    ------
    radiance : ndarray
        The small b function [1/wavelen].
    '''
    wl = change_to_quantity(wavelen, u.m, to_value=True)
    temp = change_to_quantity(temperature, u.K, to_value=True)
    norm = SIGMA_SB * temp**4
    norm_radiance = np.pi * B_lambda(wavelen=wl, temperature=temp) / norm
    return norm_radiance
