#PyQED 1.0

Uehling vacuum polarization corrections from ORCA quantum chemical calculations.

PyQED is an open-source Python post-processor that computes leading-order QED vacuum polarization (Uehling) corrections from converged ORCA 6.x calculations. It reads the SCF density matrix and basis set definitions via `orca_2json`, evaluates the screened finite-nucleus Uehling potential on a Becke-Lebedev integration grid, and returns the energy correction as a first-order perturbative estimate.

**Note:** PyQED computes only the vacuum polarization (Uehling) contribution. The electron self-energy, which is larger in magnitude but opposite in sign, is not included. For assessing whether QED matters for a given reaction energy, the VP component alone is sufficient — see the accompanying paper for details.

## Installation

```

git clone https://github.com/grishabogolyubov/pyqed.git

cd pyqed

pip install numpy scipy

```

## Usage

1. Run your ORCA calculation normally

2. Ensure `orca.json.conf` contains `{"Densities": ["scfp"], "Basisset": true, "JSONFormats": ["json"]}`

3. Convert: `orca_2json output.gbw -json`

4. Run PyQED:

```

PYTHONPATH=. python -m pyqed.cli output.json --verbose --per-atom

```

### Options

| Flag | Description |

|------|-------------|

| `--verbose` | Detailed output |

| `--per-atom` | Per-atom QED decomposition |

| `--no-screening` | Disable Welton many-electron screening |

| `--no-finite-nucleus` | Use point-nucleus Uehling potential |

## Example

```

$ PYTHONPATH=. python -m pyqed.cli Th4plus.json --verbose --per-atom

PyQED - Uehling Vacuum Polarization Correction

==================================================

Input: Th4plus.json

Atoms: 1, Basis functions: 159

Total: -9.1614e+00 Hartree

       -249.294 eV

Settings: screening=on, finite_nucleus=on

```

## Requirements

- Python 3.8+

- NumPy

- SciPy

- ORCA 6.x with `orca_2json`

## How It Works

PyQED evaluates ΔE = Tr(P · V^Ueh), where P is the SCF density matrix and V^Ueh is the Uehling potential matrix in the AO basis. The Uehling potential includes finite nuclear size (Fermi distribution) and many-electron screening (Welton model). Integration uses Mura-Knowles radial grids with Lebedev angular quadrature and Becke atomic partitioning for molecules.

## Citation

If you use PyQED, please cite:

> G. Bogolyubov, "PyQED: An Open Source Quantum Electrodynamic Correction Post-Processing Tool for Quantum Chemical Calculations" (2026).

## License

MIT

```
