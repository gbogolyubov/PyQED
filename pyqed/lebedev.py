"""
Lebedev quadrature on the unit sphere.

Provides Lebedev-Laikov quadrature rules for numerical integration over the
unit sphere.  Points and weights are generated from compact generator tables
using octahedral (Oh) symmetry operations.

Supported orders (algebraic degree of exactness) and corresponding number
of quadrature points:

    order   npts
    -----   ----
      5       14
     11       50
     17      110
     23      194
     25      230
     29      302

Reference
---------
V.I. Lebedev and D.N. Laikov,
"A quadrature formula for the sphere of the 131st algebraic order of accuracy",
Doklady Mathematics 59 (1999) 477-481.

Generator parameters are taken from the original Fortran source by Laikov:
https://server.ccl.net/cca/software/SOURCES/FORTRAN/Lebedev-Laikov-Grids/
"""

import numpy as np

__all__ = ["get_lebedev_grid", "available_orders"]

# ---------------------------------------------------------------------------
# Octahedral symmetry generators  (Oh group)
#
# These match the gen_oh subroutine from the Lebedev-Laikov Fortran code.
# Code 1:  6 pts  -- (+-1, 0, 0) and permutations
# Code 2: 12 pts  -- (0, +-a, +-a) and permutations, a = 1/sqrt(2)
# Code 3:  8 pts  -- (+-a, +-a, +-a), a = 1/sqrt(3)
# Code 4: 24 pts  -- permutations of (+-a, +-a, +-b), b = sqrt(1-2a^2)
# Code 5: 24 pts  -- (+-a, +-b, 0) and cyclic permutations, b = sqrt(1-a^2)
# Code 6: 48 pts  -- all permutations of (+-a, +-b, +-c), c = sqrt(1-a^2-b^2)
# ---------------------------------------------------------------------------


def _gen_oh(code, a=0.0, b=0.0):
    """Generate a set of points on the unit sphere under Oh symmetry.

    Parameters
    ----------
    code : int
        Symmetry code (1-6), see module docstring.
    a, b : float
        Generator parameters (used by codes 4, 5, 6).

    Returns
    -------
    pts : ndarray, shape (n, 3)
    """
    if code == 1:
        # 6 points along axes
        return np.array([
            [ 1.0,  0.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0,  1.0],
            [ 0.0,  0.0, -1.0],
        ])

    elif code == 2:
        # 12 points on face diagonals
        a = np.sqrt(0.5)
        pts = []
        for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            for sa in (a, -a):
                for sb in (a, -a):
                    p = [0.0, 0.0, 0.0]
                    p[i] = 0.0
                    p[j] = sa
                    p[k] = sb
                    pts.append(p)
        return np.array(pts)

    elif code == 3:
        # 8 points on body diagonals
        a = np.sqrt(1.0 / 3.0)
        pts = []
        for s1 in (a, -a):
            for s2 in (a, -a):
                for s3 in (a, -a):
                    pts.append([s1, s2, s3])
        return np.array(pts)

    elif code == 4:
        # 24 points from (+-a, +-a, +-b) and permutations
        b_val = np.sqrt(1.0 - 2.0 * a * a)
        pts = []
        # Three permutations: b goes in position 0, 1, or 2
        for bp in range(3):
            other = [i for i in range(3) if i != bp]
            for sa in (a, -a):
                for sb in (a, -a):
                    for sbb in (b_val, -b_val):
                        p = [0.0, 0.0, 0.0]
                        p[bp] = sbb
                        p[other[0]] = sa
                        p[other[1]] = sb
                        pts.append(p)
        return np.array(pts)

    elif code == 5:
        # 24 points: (+-a, +-b, 0), (+-b, +-a, 0) and cyclic perms of
        # the zero coordinate.  b = sqrt(1 - a^2).
        b_val = np.sqrt(1.0 - a * a)
        pts = []
        for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            for v1, v2 in [(a, b_val), (b_val, a)]:
                for s1 in (1.0, -1.0):
                    for s2 in (1.0, -1.0):
                        p = [0.0, 0.0, 0.0]
                        p[i] = s1 * v1
                        p[j] = s2 * v2
                        p[k] = 0.0
                        pts.append(p)
        return np.array(pts)

    elif code == 6:
        # 48 points from all permutations of (+-a, +-b, +-c)
        c = np.sqrt(1.0 - a * a - b * b)
        pts = []
        # All 6 permutations of (a, b, c)
        from itertools import permutations as _perms
        for perm in _perms([a, b, c]):
            for s0 in (1.0, -1.0):
                for s1 in (1.0, -1.0):
                    for s2 in (1.0, -1.0):
                        pts.append([s0 * perm[0], s1 * perm[1], s2 * perm[2]])
        return np.array(pts)

    else:
        raise ValueError(f"Unknown generator code {code}")


def _build_grid(generators):
    """Build a full Lebedev grid from generator tuples.

    Parameters
    ----------
    generators : list of (code, a, b, v)
        Each entry encodes a call to gen_oh.  *v* is the quadrature weight
        (uniform for each orbit), normalised so that the sum over all points
        equals 1.

    Returns
    -------
    points : ndarray, shape (N, 3)
    weights : ndarray, shape (N,)
        Weights scaled so that ``sum(weights) == 4 * pi``.
    """
    all_pts = []
    all_wts = []
    for code, a, b, v in generators:
        pts = _gen_oh(code, a, b)
        n = len(pts)
        all_pts.append(pts)
        all_wts.append(np.full(n, v))

    points = np.vstack(all_pts)
    weights = np.concatenate(all_wts)
    # The Lebedev-Laikov weights v_i satisfy  sum_i v_i = 1.
    # Rescale to the standard normalisation  sum_i w_i = 4 pi.
    weights = weights * (4.0 * np.pi)
    return points, weights


# ---------------------------------------------------------------------------
# Generator tables from the Lebedev-Laikov Fortran source
#
# Each entry: (code, a, b, v)
# v is the weight with sum(v_i * n_i) = 1 where n_i is the orbit size.
# ---------------------------------------------------------------------------

_GENERATORS = {}

# ---- 14 points (degree 5) ----
_GENERATORS[5] = [
    (1, 0.0, 0.0, 0.6666666666666667e-1),
    (3, 0.0, 0.0, 0.7500000000000000e-1),
]

# ---- 50 points (degree 11) ----
_GENERATORS[11] = [
    (1, 0.0, 0.0, 0.1269841269841270e-1),
    (2, 0.0, 0.0, 0.2257495590828924e-1),
    (3, 0.0, 0.0, 0.2109375000000000e-1),
    (4, 0.3015113445777636e+0, 0.0, 0.2017333553791887e-1),
]

# ---- 110 points (degree 17) ----
_GENERATORS[17] = [
    (1, 0.0, 0.0, 0.3828270494937162e-2),
    (3, 0.0, 0.0, 0.9793737512487512e-2),
    (4, 0.1851156353447362e+0, 0.0, 0.8211737283191111e-2),
    (4, 0.6904210483822922e+0, 0.0, 0.9942814891178103e-2),
    (4, 0.3956894730559419e+0, 0.0, 0.9595471336070963e-2),
    (5, 0.4783690288121502e+0, 0.0, 0.9694996361663028e-2),
]

# ---- 194 points (degree 23) ----
_GENERATORS[23] = [
    (1, 0.0, 0.0, 0.1782340447244611e-2),
    (2, 0.0, 0.0, 0.5716905949977102e-2),
    (3, 0.0, 0.0, 0.5573383178848738e-2),
    (4, 0.6712973442695226e+0, 0.0, 0.5608704082587997e-2),
    (4, 0.2892465627575439e+0, 0.0, 0.5158237711805383e-2),
    (4, 0.4446933178717437e+0, 0.0, 0.5518771467273614e-2),
    (4, 0.1299335447650067e+0, 0.0, 0.4106777028169394e-2),
    (5, 0.3457702197611283e+0, 0.0, 0.5051846064614808e-2),
    (6, 0.1590417105383530e+0, 0.8360360154824589e+0, 0.5530248916233094e-2),
]

# ---- 230 points (degree 25) ----
_GENERATORS[25] = [
    (1, 0.0, 0.0, -0.5522639919727325e-1),
    (3, 0.0, 0.0,  0.4450274607445226e-2),
    (4, 0.4492044687397611e+0, 0.0, 0.4496841067921404e-2),
    (4, 0.2520419490210201e+0, 0.0, 0.5049153450478750e-2),
    (4, 0.6981906658447242e+0, 0.0, 0.3976408018051883e-2),
    (4, 0.6587405243460960e+0, 0.0, 0.4401400650381014e-2),
    (4, 0.4038544050097660e-1, 0.0, 0.1724544350544401e-1),
    (5, 0.5823842309715585e+0, 0.0, 0.4231083095357343e-2),
    (5, 0.3545877390518688e+0, 0.0, 0.5198069864064399e-2),
    (6, 0.2272181808998187e+0, 0.4864661535886647e+0, 0.4695720972568883e-2),
]

# ---- 302 points (degree 29) ----
_GENERATORS[29] = [
    (1, 0.0, 0.0, 0.8545911725128148e-3),
    (3, 0.0, 0.0, 0.3599119285025571e-2),
    (4, 0.3515640345570105e+0, 0.0, 0.3449788424305883e-2),
    (4, 0.6566329410219612e+0, 0.0, 0.3604822601419882e-2),
    (4, 0.4729054132581005e+0, 0.0, 0.3576729661743367e-2),
    (4, 0.9618308522614784e-1, 0.0, 0.2352101413689164e-2),
    (4, 0.2219645236294178e+0, 0.0, 0.3108953122413675e-2),
    (4, 0.7011766416089545e+0, 0.0, 0.3650045807677255e-2),
    (5, 0.2644152887060663e+0, 0.0, 0.2982344963171804e-2),
    (5, 0.5718955891878961e+0, 0.0, 0.3600820932216460e-2),
    (6, 0.2510034751770465e+0, 0.8000727494073952e+0, 0.3571540554273387e-2),
    (6, 0.1233548532583327e+0, 0.4127724083168531e+0, 0.3392312205006170e-2),
]


# ---------------------------------------------------------------------------
# Map from order to expected number of points (for validation)
# ---------------------------------------------------------------------------
_EXPECTED_NPTS = {
    5: 14,
    11: 50,
    17: 110,
    23: 194,
    25: 230,
    29: 302,
}


# ---------------------------------------------------------------------------
# Build and cache grids
# ---------------------------------------------------------------------------

_GRID_CACHE = {}


def _get_or_build(order):
    """Return (points, weights) for the given order, building and caching."""
    if order not in _GRID_CACHE:
        if order not in _GENERATORS:
            raise ValueError(
                f"Lebedev grid of order {order} is not available. "
                f"Supported orders: {sorted(_GENERATORS)}"
            )
        pts, wts = _build_grid(_GENERATORS[order])
        expected = _EXPECTED_NPTS.get(order)
        if expected is not None and len(pts) != expected:
            raise RuntimeError(
                f"Lebedev grid order {order}: expected {expected} points "
                f"but generated {len(pts)}"
            )
        _GRID_CACHE[order] = (pts, wts)
    return _GRID_CACHE[order]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def available_orders():
    """Return a sorted list of supported Lebedev quadrature orders."""
    return sorted(_GENERATORS)


def get_lebedev_grid(order):
    """Return Lebedev quadrature points and weights for a given order.

    Parameters
    ----------
    order : int
        Algebraic degree of exactness.  Use :func:`available_orders` to see
        which orders are tabulated.  If *order* is not exactly available, the
        smallest available order >= *order* is used automatically.

    Returns
    -------
    points : ndarray, shape (npts, 3)
        Unit vectors (x, y, z) on the sphere.
    weights : ndarray, shape (npts,)
        Quadrature weights normalised so that ``sum(weights) == 4 * pi``.

    Raises
    ------
    ValueError
        If no tabulated order can satisfy the request.

    Examples
    --------
    >>> pts, wts = get_lebedev_grid(23)
    >>> pts.shape
    (194, 3)
    >>> abs(wts.sum() - 4 * np.pi) < 1e-12
    True
    """
    avail = available_orders()

    if order in avail:
        chosen = order
    else:
        # Pick the smallest available order >= requested
        candidates = [o for o in avail if o >= order]
        if not candidates:
            raise ValueError(
                f"Requested order {order} exceeds the largest available "
                f"order ({max(avail)}).  Supported orders: {avail}"
            )
        chosen = min(candidates)

    pts, wts = _get_or_build(chosen)
    return pts.copy(), wts.copy()
