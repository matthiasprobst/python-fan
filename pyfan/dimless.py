"""

converts physical variables of pressure and volume flow rate to dimensionless numbers psi and phi and wise versa
from dimensionless to physical variables

"""
from typing import Union

import numpy as np
from numpy import pi


def psi(dp: Union[float, np.ndarray],
        n: Union[float, np.ndarray],
        rho: Union[float, np.ndarray],
        D: float) -> Union[float, np.ndarray]:
    """Head coefficient

    Parameters
    ----------
    dp: `float` or `np.ndarray[1d]`
        pressure head [Pa]
    n: `float` or `np.ndarray[1d]`
        revolution speed [rev/s]
    rho: `float` or `np.ndarray[1d]`
        Density of fluid in [kg/m^3]
    D: `float`
        Diameter [m] of the vane
    """
    return 2 * dp / (pi ** 2 * rho * D ** 2 * n ** 2)


def psi2dptot(psi, n, rho, D):
    """Calculates pressure increase from pressure coefficient

    Parameters
    ----------
    psi: `float` or `np.ndarray[1d]`
        pressure coefficient [-]
    n: `float` or `np.ndarray[1d]`
        revolution speed [rev/s]
    rho: `float` or `np.ndarray[1d]`
        Density of fluid in [kg/m^3]
    D: `float`
        Diameter [m] of the vane
    """
    return psi * (0.5 * pi ** 2 * rho * D ** 2 * n ** 2)


def phi(vfr, n, D):
    """Flow coefficient

    Parameters
    ----------
    vfr: `float` or `np.ndarray[1d]`
        volume flow rate in [m^3/s]
    n: `float` or `np.ndarray[1d]`
        revolution speed [rev/s]
    D: `float`
        Diameter [m] of the vane
    """
    return 4 * vfr / (pi ** 2 * D ** 3 * n)


def phi2vfr(phi, n, D):
    """Calculates volume flow rate from flow coefficient

    Parameters
    ----------
    phi: `float` or `np.ndarray[1d]`
        flow coefficient [-]
    n: `float` or `np.ndarray[1d]`
        revolution speed [rev/s]
    D: `float`
        Diameter [m] of the vane
    """
    return phi * (pi ** 2 * D ** 3 * n) / 4
