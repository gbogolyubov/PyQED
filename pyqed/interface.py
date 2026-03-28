"""Read ORCA 6.x orca_2json output files."""

import json
import numpy as np
from pyqed.constants import ANGSTROM_TO_BOHR

# Number of basis functions per angular momentum
SHELL_SIZE = {"s": 1, "p": 3, "d": 5, "f": 7, "g": 9}


def read_orca_json(filepath: str) -> dict:
    """Parse orca_2json JSON and return structured data.

    Returns
    -------
    dict with keys:
        atoms : list of dict
            Each has 'coords_bohr', 'Z', 'nuclear_charge', 'element', 'basis'
            basis entries: {'shell': str, 'exponents': array, 'coefficients': array}
        density : np.ndarray, shape (nbas, nbas)
        nbas : int
        hftyp : str ('RHF' or 'UHF')
        charge : int
        multiplicity : int
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    mol = data["Molecule"]
    coord_units = mol.get("CoordinateUnits", "Angs")
    hftyp = mol.get("HFTyp", "RHF")
    charge = mol.get("Charge", 0)
    multiplicity = mol.get("Multiplicity", 1)

    scale = ANGSTROM_TO_BOHR if coord_units == "Angs" else 1.0

    atoms = []
    nbas = 0
    for atom_data in mol["Atoms"]:
        coords = np.array(atom_data["Coords"]) * scale
        z = atom_data["ElementNumber"]
        nuc_charge = atom_data.get("NuclearCharge", float(z))
        element = atom_data["ElementLabel"]

        basis_shells = []
        atom_nbas = 0
        for shell_data in atom_data["Basis"]:
            shell_type = shell_data["Shell"].lower()
            exponents = np.array(shell_data["Exponents"], dtype=np.float64)
            coefficients = np.array(shell_data["Coefficients"], dtype=np.float64)
            basis_shells.append({
                "shell": shell_type,
                "exponents": exponents,
                "coefficients": coefficients,
            })
            atom_nbas += SHELL_SIZE[shell_type]

        atoms.append({
            "coords_bohr": coords,
            "Z": z,
            "nuclear_charge": nuc_charge,
            "element": element,
            "basis": basis_shells,
            "nbas": atom_nbas,
        })
        nbas += atom_nbas

    # Extract density matrix
    densities = mol["Densities"]
    # scfp is the total density (alpha+beta for UHF)
    density_list = densities["scfp"]
    density = np.array(density_list, dtype=np.float64)

    if density.shape != (nbas, nbas):
        raise ValueError(
            f"Density matrix shape {density.shape} does not match "
            f"expected ({nbas}, {nbas})"
        )

    return {
        "atoms": atoms,
        "density": density,
        "nbas": nbas,
        "hftyp": hftyp,
        "charge": charge,
        "multiplicity": multiplicity,
    }
