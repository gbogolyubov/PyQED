"""Gaussian basis function evaluation with ORCA conventions.

Implements ORCA's real solid harmonic angular functions and
contracted Gaussian radial parts with proper normalization.
"""

import numpy as np
from math import sqrt, pi, factorial

def _double_factorial_odd(n):
    """Compute n!! for odd n: n!! = n*(n-2)*(n-4)*...*3*1. For n<=0 return 1."""
    if n <= 0:
        return 1
    result = 1
    k = n
    while k > 0:
        result *= k
        k -= 2
    return result


def primitive_norm(alpha, l):
    """Normalization constant for a primitive Gaussian r^l exp(-alpha r^2).

    Ensures that the full basis function phi = angular * R_l(r) is normalized:
    integral |angular|^2 dOmega = 1 (handled by angular part)
    integral_0^inf r^2 R_l^2(r) dr = 1 (handled here)

    where R_l(r) = N * r^l * exp(-alpha * r^2).

    This gives N^2 = 2^(2l+7/2) * alpha^(l+3/2) / ((2l+1)!! * sqrt(pi))
    """
    df = _double_factorial_odd(2 * l + 1)  # (2l+1)!!
    n_sq = (2.0 ** (2 * l + 3.5)) * (alpha ** (l + 1.5)) / (df * sqrt(pi))
    return sqrt(n_sq)


def eval_radial(r, exponents, coefficients, l):
    """Evaluate contracted radial function R_l(r) at distance r.

    R_l(r) = sum_i c_i * N_i * exp(-alpha_i * r^2)

    The r^l factor is NOT included here because it is already
    present in the angular functions (e.g., p_z = N_p * z = N_p * r * cos(theta)).

    Parameters
    ----------
    r : np.ndarray
        Radial distances from center.
    exponents : np.ndarray
        Primitive exponents alpha_i.
    coefficients : np.ndarray
        Contraction coefficients c_i.
    l : int
        Angular momentum quantum number.

    Returns
    -------
    np.ndarray
        Values of R_l at each point.
    """
    result = np.zeros_like(r)
    for alpha, coeff in zip(exponents, coefficients):
        n = primitive_norm(alpha, l)
        result += coeff * n * np.exp(-alpha * r * r)
    return result


def angular_s():
    """Return prefactor for s-type: 1/(2*sqrt(pi))."""
    return 1.0 / (2.0 * sqrt(pi))


def eval_shell_angular(shell_type, x, y, z, r2):
    """Evaluate angular part of all components for a shell.

    Uses ORCA convention from manual Section 7.58.4.2.

    Parameters
    ----------
    shell_type : str
        One of 's', 'p', 'd', 'f', 'g'.
    x, y, z : np.ndarray
        Cartesian displacements from shell center.
    r2 : np.ndarray
        r^2 = x^2 + y^2 + z^2.

    Returns
    -------
    list of np.ndarray
        Angular values for each component, in ORCA ordering.
    """
    if shell_type == "s":
        # S = 1/(2*sqrt(pi))
        pf = 1.0 / (2.0 * sqrt(pi))
        return [np.full_like(x, pf)]

    elif shell_type == "p":
        # N_p = (1/2)*sqrt(3/pi)
        np_ = 0.5 * sqrt(3.0 / pi)
        # p(0)=z, p(1)=x, p(2)=y  (ORCA: z, x, y ordering)
        return [np_ * z, np_ * x, np_ * y]

    elif shell_type == "d":
        # N_d = (1/2)*sqrt(15/pi)
        nd = 0.5 * sqrt(15.0 / pi)
        # d(0) = (sqrt(3)/6) * N_d * (3z^2 - r^2)
        # d(1) = N_d * xz
        # d(2) = N_d * yz
        # d(3) = N_d * (x^2-y^2)/2
        # d(4) = N_d * xy
        d0 = (sqrt(3.0) / 6.0) * nd * (3.0 * z * z - r2)
        d1 = nd * x * z
        d2 = nd * y * z
        d3 = nd * (x * x - y * y) / 2.0
        d4 = nd * x * y
        return [d0, d1, d2, d3, d4]

    elif shell_type == "f":
        # N_f = (1/2)*sqrt(105/pi)
        nf = 0.5 * sqrt(105.0 / pi)
        # f(0) = (sqrt(15)/30)*N_f * z*(5z^2-3r^2)
        # f(1) = (sqrt(10)/20)*N_f * x*(5z^2-r^2)
        # f(2) = (sqrt(10)/20)*N_f * y*(5z^2-r^2)
        # f(3) = (1/2)*N_f * (x^2-y^2)*z
        # f(4) = N_f * xyz
        # f(5) = -(sqrt(6)/12)*N_f * x*(x^2-3y^2)
        # f(6) = -(sqrt(6)/12)*N_f * y*(3x^2-y^2)
        f0 = (sqrt(15.0) / 30.0) * nf * z * (5.0 * z * z - 3.0 * r2)
        f1 = (sqrt(10.0) / 20.0) * nf * x * (5.0 * z * z - r2)
        f2 = (sqrt(10.0) / 20.0) * nf * y * (5.0 * z * z - r2)
        f3 = 0.5 * nf * (x * x - y * y) * z
        f4 = nf * x * y * z
        f5 = -(sqrt(6.0) / 12.0) * nf * x * (x * x - 3.0 * y * y)
        f6 = -(sqrt(6.0) / 12.0) * nf * y * (3.0 * x * x - y * y)
        return [f0, f1, f2, f3, f4, f5, f6]

    elif shell_type == "g":
        # N_g = (3/2)*sqrt(35/pi)
        ng = 1.5 * sqrt(35.0 / pi)
        x2, y2, z2 = x * x, y * y, z * z
        # g(0) = (sqrt(35)/280)*N_g*(35z^4 - 30z^2*r^2 + 3r^4)
        # g(1) = (sqrt(10)/40)*N_g*xz*(7z^2-3r^2)
        # g(2) = (sqrt(10)/40)*N_g*yz*(7z^2-3r^2)
        # g(3) = (sqrt(5)/20)*N_g*(x^2-y^2)*(7z^2-r^2)
        # g(4) = (sqrt(5)/10)*N_g*xyz*(7z^2-r^2) -- wait, standard is:
        # g(4) = (sqrt(5)/10)*N_g*xy*(7z^2-r^2)
        # g(5) = -(sqrt(70)/280)*N_g*xz*(x^2-3y^2)  -- wait
        # Let me use the standard real solid harmonic g definitions
        # consistent with ORCA manual conventions:
        r4 = r2 * r2
        g0 = (sqrt(35.0) / 280.0) * ng * (35.0 * z2 * z2 - 30.0 * z2 * r2 + 3.0 * r4)
        g1 = (sqrt(10.0) / 40.0) * ng * x * z * (7.0 * z2 - 3.0 * r2)
        g2 = (sqrt(10.0) / 40.0) * ng * y * z * (7.0 * z2 - 3.0 * r2)
        g3 = (sqrt(5.0) / 20.0) * ng * (x2 - y2) * (7.0 * z2 - r2)
        g4 = (sqrt(5.0) / 10.0) * ng * x * y * (7.0 * z2 - r2)
        g5 = -(sqrt(70.0) / 140.0) * ng * x * z * (x2 - 3.0 * y2)
        g6 = -(sqrt(70.0) / 140.0) * ng * y * z * (3.0 * x2 - y2)
        g7 = (sqrt(35.0) / 280.0) * ng * (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2))
        # g7 simplifies to (sqrt(35)/280)*N_g*(x^4 - 6x^2y^2 + y^4)
        g8 = (sqrt(35.0) / 70.0) * ng * x * y * (x2 - y2)
        return [g0, g1, g2, g3, g4, g5, g6, g7, g8]

    else:
        raise ValueError(f"Unsupported shell type: {shell_type}")


SHELL_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}


def eval_basis_functions(atoms, points):
    """Evaluate all basis functions at given points.

    Parameters
    ----------
    atoms : list of dict
        From read_orca_json, each with 'coords_bohr' and 'basis'.
    points : np.ndarray, shape (npts, 3)
        Coordinates in Bohr.

    Returns
    -------
    np.ndarray, shape (nbas, npts)
        Basis function values. Row i = basis function i evaluated at all points.
    """
    npts = points.shape[0]

    # Count total basis functions
    nbas = sum(a["nbas"] for a in atoms)
    result = np.zeros((nbas, npts), dtype=np.float64)

    ibas = 0
    for atom in atoms:
        center = atom["coords_bohr"]
        dx = points[:, 0] - center[0]
        dy = points[:, 1] - center[1]
        dz = points[:, 2] - center[2]
        r2 = dx * dx + dy * dy + dz * dz
        r = np.sqrt(r2)

        for shell in atom["basis"]:
            shell_type = shell["shell"]
            l = SHELL_L[shell_type]
            exps = shell["exponents"]
            coeffs = shell["coefficients"]

            # Radial part
            radial = eval_radial(r, exps, coeffs, l)

            # Angular parts
            angular_components = eval_shell_angular(shell_type, dx, dy, dz, r2)

            for ang in angular_components:
                result[ibas, :] = ang * radial
                ibas += 1

    assert ibas == nbas, f"Basis count mismatch: {ibas} vs {nbas}"
    return result
