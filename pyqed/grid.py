"""Numerical integration grid for QED matrix elements.

Uses Becke atomic partitioning with:
- Mura-Knowles radial mapping
- Lebedev angular grids
- Standard grid pruning
"""

import numpy as np
from pyqed.basis import eval_basis_functions
from pyqed.lebedev import get_lebedev_grid
from pyqed.uehling import uehling_point_nucleus
from pyqed.nuclear import uehling_finite_nucleus_fast
from pyqed.screening import (
    compute_electron_density_radial,
    compute_n_inner,
    screening_factor,
)
from pyqed.constants import ALPHA


def mura_knowles_radial(n, z=1):
    """Mura-Knowles radial quadrature mapping.

    Maps [0, 1] -> [0, inf) using r = -R * ln(1 - x^3)
    where R is a scaling factor depending on the row of the periodic table.

    Parameters
    ----------
    n : int
        Number of radial points.
    z : int
        Atomic number (used for scaling).

    Returns
    -------
    r : np.ndarray
        Radial points in Bohr.
    w : np.ndarray
        Integration weights (includes r^2 and Jacobian).
    """
    # Bragg-Slater radii for scaling (approximate)
    if z <= 2:
        R = 1.0
    elif z <= 10:
        R = 1.5
    elif z <= 18:
        R = 2.0
    elif z <= 36:
        R = 2.5
    elif z <= 54:
        R = 3.0
    else:
        R = 3.5

    # Uniform grid on (0, 1), avoiding endpoints
    x = np.linspace(0, 1, n + 2)[1:-1]  # n points in (0,1)

    # Mura-Knowles mapping: r = -R * ln(1 - x^3)
    r = -R * np.log(1.0 - x**3)
    # Jacobian: dr/dx = R * 3*x^2 / (1 - x^3)
    drdx = R * 3.0 * x**2 / (1.0 - x**3)
    # Weight includes r^2 for spherical integration and uniform spacing dx = 1/(n+1)
    dx = 1.0 / (n + 1)
    w = r**2 * drdx * dx

    return r, w


def becke_partition_weights(atoms, points):
    """Compute Becke atomic partition weights.

    Uses the standard Becke fuzzy cell partitioning with
    the Becke step function iterated 3 times.

    Parameters
    ----------
    atoms : list of dict
        Atomic data with 'coords_bohr' and 'Z'.
    points : np.ndarray, shape (npts, 3)
        Grid points.

    Returns
    -------
    np.ndarray, shape (natoms, npts)
        Partition weights for each atom at each point.
    """
    natoms = len(atoms)
    npts = points.shape[0]
    coords = np.array([a["coords_bohr"] for a in atoms])  # (natoms, 3)

    if natoms == 1:
        return np.ones((1, npts))

    # Compute distances from each point to each atom
    # dist[a, p] = |points[p] - coords[a]|
    dist = np.zeros((natoms, npts))
    for a in range(natoms):
        diff = points - coords[a]
        dist[a] = np.sqrt(np.sum(diff**2, axis=1))

    # Compute inter-atomic distances
    R_ab = np.zeros((natoms, natoms))
    for a in range(natoms):
        for b in range(a + 1, natoms):
            R_ab[a, b] = np.linalg.norm(coords[a] - coords[b])
            R_ab[b, a] = R_ab[a, b]

    # Becke partitioning
    # P_a(r) = product_{b != a} s(mu_ab(r))
    # mu_ab = (r_a - r_b) / R_ab
    # s(mu) = 0.5 * (1 - p(p(p(mu))))
    # p(x) = 1.5*x - 0.5*x^3

    def becke_step(mu):
        """Becke smoothing step function, iterated 3 times."""
        for _ in range(3):
            mu = 1.5 * mu - 0.5 * mu**3
        return 0.5 * (1.0 - mu)

    # Size adjustment (Becke 1988)
    # Use Bragg-Slater radii for heteronuclear adjustment
    bragg_slater = _bragg_slater_radii()
    chi = np.ones((natoms, natoms))
    for a in range(natoms):
        for b in range(natoms):
            if a != b:
                ra = bragg_slater.get(atoms[a]["Z"], 2.0)
                rb = bragg_slater.get(atoms[b]["Z"], 2.0)
                chi[a, b] = ra / rb

    # u_ab adjustment
    a_ab = np.zeros((natoms, natoms))
    for a in range(natoms):
        for b in range(natoms):
            if a != b:
                u = (chi[a, b] - 1.0) / (chi[a, b] + 1.0)
                a_ab[a, b] = u / (u**2 - 1.0) if abs(u**2 - 1.0) > 1e-14 else 0.0
                a_ab[a, b] = min(max(a_ab[a, b], -0.5), 0.5)

    P = np.ones((natoms, npts))
    for a in range(natoms):
        for b in range(natoms):
            if a == b:
                continue
            if R_ab[a, b] < 1e-14:
                continue
            mu = (dist[a] - dist[b]) / R_ab[a, b]
            nu = mu + a_ab[a, b] * (1.0 - mu**2)
            P[a] *= becke_step(nu)

    # Normalize
    P_sum = np.sum(P, axis=0)
    P_sum = np.where(P_sum < 1e-30, 1.0, P_sum)
    P /= P_sum

    return P


def _bragg_slater_radii():
    """Bragg-Slater radii in Bohr."""
    return {
        1: 0.661, 2: 0.567, 3: 2.835, 4: 1.890, 5: 1.606, 6: 1.417,
        7: 1.228, 8: 1.134, 9: 1.039, 10: 1.181, 11: 3.402, 12: 2.646,
        13: 2.457, 14: 2.079, 15: 1.890, 16: 1.890, 17: 1.795, 18: 1.984,
        19: 4.157, 20: 3.402, 26: 2.457, 29: 2.457, 30: 2.457, 35: 2.173,
        53: 2.457, 79: 2.457, 82: 2.646, 90: 3.024, 91: 3.024, 92: 2.835,
        93: 2.835, 94: 2.835,
    }


def angular_order_for_radius(r, z, base_order=25):
    """Select angular grid order based on distance from nucleus (pruning).

    Parameters
    ----------
    r : float
        Distance from nucleus in Bohr.
    z : int
        Atomic number.
    base_order : int
        Base Lebedev order.

    Returns
    -------
    int
        Lebedev order to use.
    """
    # Pruning: use fewer angular points very near and very far from nucleus
    if r < 0.01:
        return 6   # 14 points
    elif r < 0.1:
        return 17  # 110 points
    elif r > 15.0:
        return 11  # 50 points
    else:
        return base_order


def compute_qed_correction(
    atoms,
    density_matrix,
    radial_points=175,
    angular_order=25,
    use_screening=True,
    use_finite_nucleus=True,
    verbose=False,
):
    """Compute QED Uehling vacuum polarization correction.

    ΔE_QED = Tr(P · V_Ueh)

    where V_Ueh is the Uehling potential matrix in the AO basis.

    Parameters
    ----------
    atoms : list of dict
        Atomic data from read_orca_json.
    density_matrix : np.ndarray, shape (nbas, nbas)
        Total electron density matrix.
    radial_points : int
        Number of radial grid points per atom.
    angular_order : int
        Base Lebedev angular grid order.
    use_screening : bool
        Whether to apply many-electron screening.
    use_finite_nucleus : bool
        Whether to use finite nuclear size.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict
        'total_hartree': float, total QED correction
        'per_atom': list of float, per-atom contributions
        'atoms': list of str, atom labels
    """
    natoms = len(atoms)
    nbas = sum(a["nbas"] for a in atoms)
    per_atom = []

    if verbose:
        print(f"Computing QED correction with {natoms} atoms, {nbas} basis functions")
        print(f"Grid: {radial_points} radial x order-{angular_order} angular")
        print(f"Screening: {use_screening}, Finite nucleus: {use_finite_nucleus}")

    # Pre-compute screening data if needed
    screening_data = {}
    if use_screening:
        if verbose:
            print("Computing electron density for screening...")
        for ia in range(natoms):
            z = atoms[ia]["Z"]
            # Radial grid for screening calculation
            r_scr, _ = mura_knowles_radial(100, z)
            r_scr = np.sort(r_scr)
            rho_e = compute_electron_density_radial(
                atoms, density_matrix, r_scr, ia
            )
            n_inner = compute_n_inner(r_scr, rho_e)
            screening_data[ia] = (r_scr, n_inner)

    for ia in range(natoms):
        z = atoms[ia]["Z"]
        center = atoms[ia]["coords_bohr"]

        if verbose:
            print(f"  Atom {ia}: {atoms[ia]['element']} (Z={z})")

        # Generate atom-centered grid
        r_rad, w_rad = mura_knowles_radial(radial_points, z)

        # For this atom, compute V_Ueh on the radial grid first
        if use_finite_nucleus:
            v_ueh_radial = uehling_finite_nucleus_fast(r_rad, z, npts_nuc=80)
        else:
            v_ueh_radial = uehling_point_nucleus(r_rad, z)

        # Apply screening if requested
        if use_screening and ia in screening_data:
            r_scr, n_inner = screening_data[ia]
            scr_factor = screening_factor(r_scr, n_inner, z)
            # Interpolate screening factor to the integration grid
            scr_on_grid = np.interp(r_rad, r_scr, scr_factor, left=1.0, right=0.0)
            v_ueh_radial *= scr_on_grid

        # Now build the contribution: for each radial shell, evaluate
        # basis functions and accumulate Tr(P * V) contribution
        atom_energy = 0.0

        # Group radial points by angular order (for pruning)
        for ir in range(len(r_rad)):
            r = r_rad[ir]
            wr = w_rad[ir]

            if abs(v_ueh_radial[ir]) < 1e-30:
                continue

            # Get angular grid (with pruning)
            ang_order = angular_order_for_radius(r, z, angular_order)
            ang_pts, ang_wts = get_lebedev_grid(ang_order)
            nang = len(ang_wts)

            # 3D points: center + r * angular_points
            pts_3d = center[np.newaxis, :] + r * ang_pts  # (nang, 3)

            # Evaluate all basis functions at these points
            bfs = eval_basis_functions(atoms, pts_3d)  # (nbas, nang)

            # The potential is V_Ueh(r) for all points at this radius
            # (since they're all at distance r from this atom)
            v = v_ueh_radial[ir]

            # For multi-atom: need Becke partition weights
            if natoms > 1:
                becke_w = becke_partition_weights(atoms, pts_3d)  # (natoms, nang)
                part_w = becke_w[ia]  # (nang,)
            else:
                part_w = np.ones(nang)

            # Contribution to Tr(P * V_Ueh):
            # sum_ang w_ang * w_rad * V(r) * becke * sum_{mu,nu} P_{mu,nu} phi_mu phi_nu
            # = w_rad * V(r) * sum_ang w_ang * becke * rho(r)
            # But we need the matrix element form:
            # V_{mu,nu} = int phi_mu(r) V(r) phi_nu(r) dr
            # E = Tr(P * V) = sum_{mu,nu} P_{mu,nu} V_{mu,nu}
            # = sum_{mu,nu} P_{mu,nu} * sum_grid w_grid * phi_mu * V * phi_nu

            # rho at each angular point: sum_{mu,nu} P_{mu,nu} phi_mu phi_nu
            # = diag(bfs^T P bfs)
            # For efficiency: rho = einsum("ip,ij,jp->p", bfs, P, bfs)
            # But since V is the same for all points at this shell:
            rho_pts = np.einsum("ip,ij,jp->p", bfs, density_matrix, bfs)

            # Accumulate energy
            atom_energy += wr * v * np.sum(ang_wts * part_w * rho_pts)

        per_atom.append(atom_energy)
        if verbose:
            print(f"    ΔE = {atom_energy:.10e} Hartree")

    total = sum(per_atom)

    return {
        "total_hartree": total,
        "per_atom": per_atom,
        "atoms": [f"{a['element']}{i}" for i, a in enumerate(atoms)],
    }
