import numpy as np
from astropy import units as u

__all__ = ["PI", "D2R", "R2D", "AU", "MICRON", "TIU", "NOUNIT",
           "GG"      , "GG_U"    , "GG_Q"     ,
           "HH"      , "HH_U"    , "HH_Q"     ,
           "KB"      , "KB_U"    , "KB_Q"     ,
           "SIGMA_SB", "SIGMA_SB", "SIGMA_SB" ,
           "CC"      , "CC"      , "CC"       ,
           "R_SUN"   , "R_SUN"   , "R_SUN"    ,
           "T_SUN"   , "T_SUN"   , "T_SUN"    ,
           "L_SUN"   , "L_SUN"   , "L_SUN",
           "C_F_SUN" , "C_F_THER"
           ]

# define constants (All in SI unit!!!)
PI = np.pi
D2R = PI/180
R2D = 180/PI
AU = 149597870700.0       # [m] 1 au in SI
MICRON = 1.e-6            # [m] 1 um in SI
TIU = u.def_unit("tiu", u.J/u.K/u.m**2/u.s**0.5)  # thermal inertia unit
NOUNIT = u.dimensionless_unscaled

# [m^3/kg/s^2] Gravitational constant
GG   = 6.67430e-11
GG_U = u.m**3/u.kg/u.s**2
GG_Q = GG*GG_U
# GG_Q = {True: GG*GG_U, False: GG}

# [J.s] Planck constant
HH = 6.62607004e-34
HH_U = u.J*u.s
HH_Q = HH*HH_U
# HH_Q = {True: HH*HH_U, False: HH}

# [J/K] Boltzman constant
KB   = 1.38064852e-23
KB_U = u.J/u.K
KB_Q = KB*KB_U
# KB_Q = {True: KB*KB_U, False: KB}

# [W/m^2/K^4] Stefan--Boltzmann constant
SIGMA_SB   = 5.670367e-08
SIGMA_SB_U = u.W/u.m**2/u.K**4
SIGMA_SB_Q = SIGMA_SB*SIGMA_SB_U
# SIGMA_SB_Q = {True: SIGMA_SB*SIGMA_SB_U, False: SIGMA_SB}

# [m/s] Speed of light
CC   = 299792458.0
CC_U = u.m/u.s
CC_Q = CC * CC_U
# CC_Q = {True: CC*CC_U, False: CC}

# [m] solar radius
R_SUN   = 695700000.0
R_SUN_U = u.m
R_SUN_Q = R_SUN*R_SUN_U
# R_SUN_Q = {True: R_SUN*R_SUN_U, False: R_SUN}

# [K] "effective" temperature of the Sun
T_SUN   = 5777
T_SUN_U = u.K
T_SUN_Q = T_SUN*T_SUN_U
# T_SUN_Q = {True: T_SUN*T_SUN_U, False: T_SUN}

# [W] solar luminosity
L_SUN   = 3.838e+26
L_SUN_U = u.W
L_SUN_Q = L_SUN*L_SUN_U
# L_SUN_Q = {True: L_SUN*L_SUN_U, False: L_SUN}

S1AU = 1361.2

C_F_SUN = (PI * SIGMA_SB / CC) * (R_SUN**2 * T_SUN**4) * (MICRON/AU)**2
# ~ 1.4313e-17 in SI [N].
C_F_THER = (PI * SIGMA_SB / CC) * (MICRON)**2
# ~ 5.9421e-28 in SI [N].
C_A_SUN = (3 * SIGMA_SB / CC) * (4 * R_SUN**2 * T_SUN**4) * (MICRON / AU**2)
# C_A_THER =
# a in um, r_h in au, rho(mass_den) in kg/m3
