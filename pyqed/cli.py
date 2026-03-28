"""Command-line interface for PyQED."""

import argparse
import sys
import time

from pyqed.interface import read_orca_json
from pyqed.grid import compute_qed_correction
from pyqed.constants import HARTREE_TO_EV, HARTREE_TO_KCAL, HARTREE_TO_CM


def main():
    parser = argparse.ArgumentParser(
        prog="pyqed",
        description="Compute Uehling vacuum polarization QED corrections from ORCA calculations.",
    )
    parser.add_argument("input", help="ORCA orca_2json output file (JSON)")
    parser.add_argument(
        "--grid-radial", type=int, default=175,
        help="Radial grid points per atom (default: 175)",
    )
    parser.add_argument(
        "--grid-angular", type=int, default=25,
        help="Lebedev angular grid order (default: 25)",
    )
    parser.add_argument(
        "--no-screening", action="store_true",
        help="Disable many-electron screening",
    )
    parser.add_argument(
        "--no-finite-nucleus", action="store_true",
        help="Use point nucleus approximation",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed output",
    )
    parser.add_argument(
        "--per-atom", action="store_true",
        help="Print per-atom QED decomposition",
    )

    args = parser.parse_args()

    # Read ORCA output
    try:
        data = read_orca_json(args.input)
    except Exception as e:
        print(f"Error reading {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"PyQED - Uehling Vacuum Polarization Correction")
        print(f"=" * 50)
        print(f"Input: {args.input}")
        print(f"HF type: {data['hftyp']}")
        print(f"Charge: {data['charge']}, Multiplicity: {data['multiplicity']}")
        print(f"Atoms: {len(data['atoms'])}, Basis functions: {data['nbas']}")
        print()

    t0 = time.time()

    result = compute_qed_correction(
        atoms=data["atoms"],
        density_matrix=data["density"],
        radial_points=args.grid_radial,
        angular_order=args.grid_angular,
        use_screening=not args.no_screening,
        use_finite_nucleus=not args.no_finite_nucleus,
        verbose=args.verbose,
    )

    elapsed = time.time() - t0

    # Output
    E = result["total_hartree"]
    print()
    print(f"QED Uehling Vacuum Polarization Correction")
    print(f"-" * 45)
    print(f"  Total: {E:+.10e} Hartree")
    print(f"         {E * HARTREE_TO_EV:+.6f} eV")
    print(f"         {E * HARTREE_TO_KCAL:+.6f} kcal/mol")
    print(f"         {E * HARTREE_TO_CM:+.2f} cm^-1")

    if args.per_atom:
        print()
        print(f"Per-atom decomposition:")
        print(f"  {'Atom':<10} {'Hartree':>16} {'eV':>12} {'%':>8}")
        print(f"  {'-'*46}")
        for label, e_atom in zip(result["atoms"], result["per_atom"]):
            pct = 100.0 * e_atom / E if abs(E) > 1e-30 else 0.0
            print(f"  {label:<10} {e_atom:>+16.10e} {e_atom * HARTREE_TO_EV:>+12.6f} {pct:>7.1f}%")

    print()
    print(f"Settings: screening={'on' if not args.no_screening else 'off'}, "
          f"finite_nucleus={'on' if not args.no_finite_nucleus else 'off'}")
    print(f"Grid: {args.grid_radial} radial x order-{args.grid_angular} angular")
    print(f"Wall time: {elapsed:.1f} s")
    print()
    print(f"Context: typical basis set error ~10-100 kcal/mol")


if __name__ == "__main__":
    main()
