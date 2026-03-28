"""Finite nuclear size corrections via Fermi charge distribution.

Fermi distribution: rho(r) = rho_0 / (1 + exp((r-c)/a))
where:
  a = 0.523 fm (skin thickness)
  c = sqrt(5/3 * <r^2> - 7*pi^2*a^2/3) (half-density radius)

The finite-nucleus Uehling potential is obtained by convolving the
point-nucleus Uehling potential with the nuclear charge distribution.
"""

import numpy as np
from scipy import integrate
from pyqed.constants import (
    ALPHA, FERMI_A_BOHR, get_rms_radius_bohr, fermi_c_from_rms
)
from pyqed.uehling import uehling_point_nucleus


def fermi_density(r, c, a=FERMI_A_BOHR):
    """Fermi charge distribution (unnormalized).

    Parameters
    ----------
    r : np.ndarray
        Radial distances in Bohr.
    c : float
        Half-density radius in Bohr.
    a : float
        Skin thickness parameter in Bohr.

    Returns
    -------
    np.ndarray
        Unnormalized density values.
    """
    # Avoid overflow in exp for large (r-c)/a
    arg = (r - c) / a
    # Clip to avoid overflow
    arg = np.clip(arg, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(arg))


def fermi_normalization(c, a=FERMI_A_BOHR, npts=500):
    """Compute normalization integral: integral 4*pi*r^2*rho(r) dr = Z (set to 1).

    Uses Gauss-Legendre quadrature on [0, rmax] where rmax = c + 20*a.
    """
    rmax = c + 30.0 * a
    nodes, weights = np.polynomial.legendre.leggauss(npts)
    # Map from [-1,1] to [0, rmax]
    r = 0.5 * rmax * (nodes + 1.0)
    w = 0.5 * rmax * weights
    rho = fermi_density(r, c, a)
    return 4.0 * np.pi * np.sum(w * r * r * rho)


def uehling_finite_nucleus(r_eval, z, npts_nuc=200):
    """Compute finite-nucleus Uehling potential by convolution.

    V_Ueh^finite(r) = integral rho_nuc(r') * V_Ueh^point(|r - r'|) d^3r'
                    = integral_0^inf 4*pi*r'^2 * rho_norm(r') *
                      V_Ueh_shell(r, r') dr'

    where V_Ueh_shell is the angular average of V_Ueh^point over a shell:
    V_Ueh_shell(r, r') = (1/(2*r*r')) * integral_{|r-r'|}^{r+r'}
                          s * V_Ueh^point(s, Z=1) ds

    But for simplicity and accuracy, we directly integrate using:
    V_Ueh^finite(r) = (4*pi / r) * integral_0^inf r'^2 rho_norm(r') *
                       [ integral_{|r-r'|}^{r+r'} s * V_Ueh^point(s, Z=1) / (2*r'*r) ds ] dr'

    Actually, the cleanest approach: for each evaluation point r, integrate
    the convolution on a radial grid.

    For spherically symmetric nuclear density:
    V_Ueh^finite(r) = integral_0^inf 4*pi*r'^2 * rho_norm(r') *
                       <V_Ueh^point(|r-r'|)>_angle dr'

    The angular average of V(|r-r'|) over r' directions:
    <V(|r-r'|)>_angle = 1/(2*r*r') * integral_{|r-r'|}^{r+r'} s * V(s) ds
    for r > 0, r' > 0.

    Parameters
    ----------
    r_eval : float or np.ndarray
        Distance(s) from nucleus center in Bohr.
    z : int
        Atomic number.
    npts_nuc : int
        Number of quadrature points for nuclear integration.

    Returns
    -------
    float or np.ndarray
        Finite-nucleus Uehling potential in Hartree.
    """
    scalar = np.isscalar(r_eval)
    r_eval = np.atleast_1d(np.asarray(r_eval, dtype=np.float64))

    rms = get_rms_radius_bohr(z)
    c = fermi_c_from_rms(rms)

    if c < 1e-20:
        # Point nucleus (very light element with tiny nucleus)
        result = uehling_point_nucleus(r_eval, z)
        return result[0] if scalar else result

    a = FERMI_A_BOHR

    # Normalization
    norm = fermi_normalization(c, a, npts=npts_nuc)

    # Nuclear integration grid: [0, rmax_nuc]
    rmax_nuc = c + 20.0 * a
    # Use Gauss-Legendre
    nodes, weights = np.polynomial.legendre.leggauss(npts_nuc)
    r_nuc = 0.5 * rmax_nuc * (nodes + 1.0)
    w_nuc = 0.5 * rmax_nuc * weights

    # Nuclear density on grid (normalized to Z)
    rho_nuc = z * fermi_density(r_nuc, c, a) / norm

    result = np.zeros_like(r_eval)

    for i, r in enumerate(r_eval):
        if r < 1e-20:
            # At r=0: V = integral 4*pi*r'^2 * rho(r') * V_point(r') dr'
            # (angular average at r=0 is just V(r') itself)
            v_at_rprime = uehling_point_nucleus(r_nuc, 1.0)  # Z=1, will multiply by Z via rho
            # Actually rho already has Z in it, and V_point has Z too
            # We need V_point with Z=1, then multiply by rho which has Z
            # Wait: V_Ueh^point ~ Z, so convolution with normalized rho gives Z * V
            # But rho_nuc already normalized to Z, so use V_point(Z=1)
            integrand = 4.0 * np.pi * r_nuc**2 * (rho_nuc / z) * \
                        uehling_point_nucleus(r_nuc, z)
            result[i] = np.sum(w_nuc * integrand)
            continue

        # For each r' on the nuclear grid, compute angular-averaged potential
        val = 0.0
        for j, rp in enumerate(r_nuc):
            if rp < 1e-20:
                continue
            # Angular average of V_Ueh^point(|r-r'|) over r' directions
            # = 1/(2*r*rp) * integral_{|r-rp|}^{r+rp} s * V(s) ds
            s_min = abs(r - rp)
            s_max = r + rp

            # Use a few-point quadrature for the s integral
            ns = 20
            s_nodes, s_weights = np.polynomial.legendre.leggauss(ns)
            s_pts = 0.5 * (s_max - s_min) * (s_nodes + 1.0) + s_min
            s_w = 0.5 * (s_max - s_min) * s_weights

            v_s = uehling_point_nucleus(s_pts, z)
            ang_avg = np.sum(s_w * s_pts * v_s) / (2.0 * r * rp)

            val += w_nuc[j] * 4.0 * np.pi * rp**2 * (rho_nuc[j] / z) * ang_avg

        result[i] = val

    if scalar:
        return result[0]
    return result


def uehling_finite_nucleus_fast(r_eval, z, npts_nuc=100):
    """Fast finite-nucleus Uehling using direct radial convolution.

    Instead of full angular averaging, uses the identity that for
    a spherical charge distribution, the electrostatic-like convolution
    simplifies for potentials that depend only on distance.

    This uses a simpler approach: evaluate V_Ueh^point on a fine radial
    grid, then convolve with the nuclear charge distribution.

    For the Uehling potential (which is short-ranged ~ exp(-2r/alpha)),
    the nuclear size correction is small for light nuclei and only
    significant for heavy nuclei where the nuclear radius is comparable
    to the Compton wavelength.

    Parameters
    ----------
    r_eval : np.ndarray
        Evaluation distances in Bohr.
    z : int
        Atomic number.
    npts_nuc : int
        Nuclear quadrature points.

    Returns
    -------
    np.ndarray
        Finite-nucleus Uehling potential.
    """
    scalar = np.isscalar(r_eval)
    r_eval = np.atleast_1d(np.asarray(r_eval, dtype=np.float64))

    rms = get_rms_radius_bohr(z)
    c = fermi_c_from_rms(rms)

    if c < 1e-20:
        result = uehling_point_nucleus(r_eval, z)
        return result[0] if scalar else result

    a = FERMI_A_BOHR
    norm = fermi_normalization(c, a, npts=300)

    # For each eval point, do a 1D convolution integral
    rmax_nuc = c + 20.0 * a
    nodes, weights = np.polynomial.legendre.leggauss(npts_nuc)
    r_nuc = 0.5 * rmax_nuc * (nodes + 1.0)
    w_nuc = 0.5 * rmax_nuc * weights

    rho_nuc_normalized = fermi_density(r_nuc, c, a) / norm  # integrates to 1

    result = np.zeros_like(r_eval)

    for i, r in enumerate(r_eval):
        # V_finite(r) = integral_0^inf 4pi r'^2 rho(r')/Z *
        #   [1/(2rr') integral_{|r-r'|}^{r+r'} s V_point(s,Z) ds] dr'
        # For r, r' both in nuclear region (very small), we can use
        # a simpler approach since the Uehling potential varies slowly
        # over the nuclear volume for light elements.

        # For heavy elements, need full convolution:
        if r < 1e-20:
            # V(0) = integral 4pi r'^2 rho(r') V_point(r', Z) dr' / Z
            v_pts = uehling_point_nucleus(r_nuc, z)
            result[i] = np.sum(w_nuc * 4.0 * np.pi * r_nuc**2 * rho_nuc_normalized * v_pts)
        else:
            val = 0.0
            for j in range(len(r_nuc)):
                rp = r_nuc[j]
                if rp < 1e-25:
                    continue
                s_min = abs(r - rp)
                s_max = r + rp
                # Quick 10-point quadrature for s integral
                ns = 16
                sn, sw = np.polynomial.legendre.leggauss(ns)
                s_pts = 0.5 * (s_max - s_min) * (sn + 1.0) + s_min
                s_w = 0.5 * (s_max - s_min) * sw
                v_s = uehling_point_nucleus(s_pts, z)
                ang_avg = np.sum(s_w * s_pts * v_s) / (2.0 * r * rp)
                val += w_nuc[j] * 4.0 * np.pi * rp**2 * rho_nuc_normalized[j] * ang_avg
            result[i] = val

    if scalar:
        return result[0]
    return result
