"""Many-electron screening of the Uehling potential.

V_scr(r) = V_Ueh^finite(r) * (Z - N_inner(r)) / Z

where N_inner(r) = integral_0^r 4*pi*r'^2 * rho_e(r') dr'
and rho_e(r) is the spherically-averaged electron density.
"""

import numpy as np
from pyqed.basis import eval_basis_functions


def compute_electron_density_radial(atoms, density_matrix, r_grid, atom_idx):
    """Compute spherically-averaged electron density around a specific atom.

    Evaluates the electron density rho(r) = sum_{mu,nu} P_{mu,nu} phi_mu(r) phi_nu(r)
    on a set of points at various distances from the atom, averaging over angles
    using a small Lebedev grid.

    Parameters
    ----------
    atoms : list of dict
        Atomic data from read_orca_json.
    density_matrix : np.ndarray, shape (nbas, nbas)
        Total electron density matrix.
    r_grid : np.ndarray
        Radial distances from the atom in Bohr.
    atom_idx : int
        Index of the atom to center on.

    Returns
    -------
    np.ndarray
        Spherically-averaged electron density at each radial point.
    """
    from pyqed.lebedev import get_lebedev_grid

    center = atoms[atom_idx]["coords_bohr"]
    # Use a modest angular grid for spherical averaging
    ang_pts, ang_wts = get_lebedev_grid(11)  # 50 points
    ang_wts_norm = ang_wts / (4.0 * np.pi)  # normalize to integrate to 1

    rho_avg = np.zeros(len(r_grid))

    for ir, r in enumerate(r_grid):
        if r < 1e-20:
            # At r=0, place a single point at the center
            pts = center.reshape(1, 3)
            bfs = eval_basis_functions(atoms, pts)  # (nbas, 1)
            rho_avg[ir] = np.einsum("i,ij,j->", bfs[:, 0], density_matrix, bfs[:, 0])
            continue

        # Generate 3D points: center + r * angular_points
        pts = center[np.newaxis, :] + r * ang_pts  # (nang, 3)
        bfs = eval_basis_functions(atoms, pts)  # (nbas, nang)

        # rho at each angular point
        rho_angular = np.einsum("ip,ij,jp->p", bfs, density_matrix, bfs)

        # Spherical average
        rho_avg[ir] = np.sum(ang_wts_norm * rho_angular)

    return rho_avg


def compute_n_inner(r_grid, rho_e):
    """Compute cumulative electron count N_inner(r) = integral_0^r 4*pi*r'^2*rho_e(r') dr'.

    Uses trapezoidal integration on the provided radial grid.

    Parameters
    ----------
    r_grid : np.ndarray
        Radial distances (sorted, ascending).
    rho_e : np.ndarray
        Spherically-averaged electron density at each grid point.

    Returns
    -------
    np.ndarray
        Cumulative electron count at each radial distance.
    """
    integrand = 4.0 * np.pi * r_grid**2 * rho_e
    n_inner = np.zeros_like(r_grid)
    for i in range(1, len(r_grid)):
        dr = r_grid[i] - r_grid[i - 1]
        n_inner[i] = n_inner[i - 1] + 0.5 * dr * (integrand[i - 1] + integrand[i])
    return n_inner


def screening_factor(r_grid, n_inner, z):
    """Compute screening factor (Z - N_inner(r)) / Z.

    Parameters
    ----------
    r_grid : np.ndarray
        Radial distances.
    n_inner : np.ndarray
        Cumulative electron count.
    z : int
        Nuclear charge.

    Returns
    -------
    np.ndarray
        Screening factor, clipped to [0, 1].
    """
    factor = (z - n_inner) / z
    return np.clip(factor, 0.0, 1.0)
