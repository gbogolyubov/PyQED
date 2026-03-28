"""Uehling vacuum polarization potential.

Point-nucleus Uehling potential (Eq. 1):
V_Ueh(r) = -(2*alpha*Z)/(3*pi*r) * integral_1^inf exp(-2*r*t/(alpha)) *
            (1 + 1/(2*t^2)) * sqrt(t^2-1)/t^2 dt

Uses substitution u = sqrt(t-1) to remove integrable singularity at t=1.
"""

import numpy as np
from scipy import integrate
from pyqed.constants import ALPHA


def _uehling_integrand_u(u, r_over_compton):
    """Uehling integrand after substitution u = sqrt(t-1).

    t = u^2 + 1, dt = 2u du
    sqrt(t^2-1) = u*sqrt(u^2+2)

    Full integrand in u:
    exp(-2R(u^2+1)) * (1 + 1/(2(u^2+1)^2)) * u*sqrt(u^2+2) / (u^2+1)^2 * 2u
    """
    t = u * u + 1.0
    t2 = t * t
    exp_val = np.exp(-2.0 * r_over_compton * t)
    factor1 = 1.0 + 0.5 / t2
    sqrt_part = u * np.sqrt(u * u + 2.0)
    return exp_val * factor1 * sqrt_part / t2 * 2.0 * u


def uehling_point_nucleus(r, z):
    """Compute point-nucleus Uehling potential at distance r from nucleus.

    Parameters
    ----------
    r : float or np.ndarray
        Distance from nucleus in Bohr.
    z : int or float
        Nuclear charge.

    Returns
    -------
    float or np.ndarray
        Uehling potential in Hartree.
    """
    scalar = np.isscalar(r)
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    result = np.zeros_like(r)

    prefactor = -2.0 * ALPHA * z / (3.0 * np.pi)

    for i, ri in enumerate(r):
        if ri < 1e-20:
            result[i] = 0.0
            continue

        R = ri / ALPHA  # r / Compton wavelength in a.u.
        u_max = np.sqrt(19.0) if R >= 0.1 else np.sqrt(99.0)

        val, _ = integrate.quad(
            _uehling_integrand_u, 0.0, u_max,
            args=(R,),
            limit=200, epsabs=1e-15, epsrel=1e-12,
        )
        result[i] = prefactor / ri * val

    if scalar:
        return result[0]
    return result
