"""PyQED: Uehling vacuum polarization corrections from ORCA calculations."""

__version__ = "0.1.0"

from pyqed.interface import read_orca_json
from pyqed.grid import compute_qed_correction
