"""Physical constants and nuclear data for QED calculations.

All values in atomic units unless noted otherwise.
"""

import math

# Fine structure constant (CODATA 2018)
ALPHA = 1.0 / 137.035999084

# Bohr radius in meters
BOHR_SI = 5.29177210903e-11

# Conversion factors
ANGSTROM_TO_BOHR = 1.8897259886
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_CM = 219474.63136320

# Electron Compton wavelength / (2 pi) in Bohr = alpha (in atomic units)
COMPTON_BOHR = ALPHA  # ħ/(m_e c) in a.u. = α a₀, but in a.u. a₀=1, so = α

# Fermi to Bohr conversion: 1 fm = 1e-15 m / BOHR_SI m
FM_TO_BOHR = 1.0e-15 / BOHR_SI

# Fermi distribution skin thickness parameter
# a = t / (4 ln 3), with t = 2.30 fm (skin thickness)
# Commonly a = 0.523 fm
FERMI_A_FM = 0.523
FERMI_A_BOHR = FERMI_A_FM * FM_TO_BOHR

# RMS charge radii in fm from Angeli & Marinova 2013 (Table I)
# Format: Z -> rms radius in fm
RMS_CHARGE_RADII_FM = {
    1:  0.8783,   # H
    2:  1.6755,   # He
    3:  2.5890,   # Li
    4:  2.5190,   # Be
    5:  2.4277,   # B
    6:  2.4702,   # C
    7:  2.5582,   # N
    8:  2.6991,   # O
    9:  2.8976,   # F
    10: 3.0055,   # Ne
    11: 2.9936,   # Na
    12: 3.0570,   # Mg
    13: 3.0610,   # Al
    14: 3.1224,   # Si
    15: 3.1889,   # P
    16: 3.2611,   # S
    17: 3.3654,   # Cl
    18: 3.4274,   # Ar
    19: 3.4349,   # K
    20: 3.4776,   # Ca
    26: 3.7377,   # Fe
    29: 3.8823,   # Cu
    30: 3.9283,   # Zn
    35: 4.1629,   # Br
    47: 4.4865,   # Ag
    53: 4.7500,   # I
    79: 5.4371,   # Au
    82: 5.5012,   # Pb
    86: 5.5903,   # Rn
    88: 5.6688,   # Ra
    89: 5.6700,   # Ac
    90: 5.7848,   # Th
    91: 5.7880,   # Pa
    92: 5.8571,   # U
    93: 5.8380,   # Np
    94: 5.8601,   # Pu
}


def get_rms_radius_bohr(z: int) -> float:
    """Get RMS charge radius in Bohr for element with atomic number z."""
    if z in RMS_CHARGE_RADII_FM:
        return RMS_CHARGE_RADII_FM[z] * FM_TO_BOHR
    # Empirical formula: r ≈ 0.836 * A^(1/3) + 0.570 fm
    # Use A ≈ 2.5*Z for heavy elements (rough)
    a_approx = 2.5 * z if z > 20 else 2.0 * z
    r_fm = 0.836 * a_approx ** (1.0 / 3.0) + 0.570
    return r_fm * FM_TO_BOHR


def fermi_c_from_rms(rms_bohr: float) -> float:
    """Compute Fermi distribution half-density radius c from RMS radius.

    c = sqrt(5/3 * <r^2> - 7*pi^2*a^2/3)
    """
    a = FERMI_A_BOHR
    arg = 5.0 / 3.0 * rms_bohr**2 - 7.0 * math.pi**2 * a**2 / 3.0
    if arg < 0:
        # For very light nuclei, use point nucleus approximation
        return 0.0
    return math.sqrt(arg)
