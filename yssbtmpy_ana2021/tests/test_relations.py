from itertools import product

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose as allclose

from ..constants import NOUNIT, PI, S1AU, SIGMA_SB, TIU
from ..relations import (AG2p, DH2p, G2q, T_eqm, pA2G, pD2H, pG2A, pH2D, q2G,
                         solve_Gq, solve_pAG, solve_pDH, solve_temp_eqm,
                         solve_thermal_par)

Qo = dict(return_quantity=True)
Qx = dict(return_quantity=False)
CoQo = dict(classical=True, return_quantity=True)
CoQx = dict(classical=True, return_quantity=False)
CxQo = dict(classical=False, return_quantity=True)
CxQx = dict(classical=False, return_quantity=False)

_test_p = [0.01, 0.1, 0.2, 1.0]  # NOUNIT
_test_G = [0.01, 0.15, 1.0]  # NOUNIT
_test_q = [0.3, 0.5, 1.0]  # NOUNIT
_test_A = [0.01, 0.1, 0.3, 1.0]  # NOUNIT
_test_H = [0, 15]  # NOUNIT (or u.mag, but not used in this package)
_test_D = [1, 1329]  # u.km
_test_d0 = [1329, 1000, 2000]  # u.km
_test_eps = [0.1, 0.9, 1.0]  # NOUNIT
_test_beam = [0.7, 1.0, 3.0]  # NOUNIT
_test_rh = [0.1, 1.0, 3.0]  # u.au
_test_T = [100, 300, 1000]  # u.K
_test_P = [0.1, 1, 10]  # u.h
_test_TI = [1, 100, 1000]  # TIU


def allclose_Q(a, b, rtol=1.e-5, atol=None, **kwargs):
    ''' astropy quantity allclose, but with ``assert`` version for testing.
    '''
    assert u.allclose(a, b, rtol=rtol, atol=atol, **kwargs)


# Test G(slope_par) and q(phase_int)
def test_G2q_CoQo():
    ''' Tests solve_Gq and G2q with CoQo
    '''
    kwdict = CoQo
    test_G = np.array(_test_G)
    real_q = 0.290 + 0.684*test_G
    calc_q1 = G2q(test_G, **kwdict)
    calc_q2 = solve_Gq(test_G, None, **kwdict)["phase_int"]

    allclose_Q(calc_q1, real_q*NOUNIT)
    allclose_Q(calc_q2, real_q*NOUNIT)


def test_G2q_CoQx():
    ''' Tests solve_Gq and G2q with CoQx
    '''
    kwdict = CoQx
    test_G = np.array(_test_G)
    real_q = 0.290 + 0.684*test_G
    calc_q1 = G2q(test_G, **kwdict)
    calc_q2 = solve_Gq(test_G, None, **kwdict)["phase_int"]

    allclose(calc_q1, real_q)
    allclose(calc_q2, real_q)


def test_G2q_CxQo():
    ''' Tests solve_Gq and G2q with CxQo
    '''
    kwdict = CxQo
    test_G = np.array(_test_G)
    real_q = 0.286 + 0.656*test_G
    calc_q1 = G2q(test_G, **kwdict)
    calc_q2 = solve_Gq(test_G, None, **kwdict)["phase_int"]

    allclose_Q(calc_q1, real_q*NOUNIT)
    allclose_Q(calc_q2, real_q*NOUNIT)


def test_G2q_CxQx():
    ''' Tests solve_Gq and G2q with CxQx
    '''
    kwdict = CxQx
    test_G = np.array(_test_G)
    real_q = 0.286 + 0.656*test_G
    calc_q1 = G2q(test_G, **kwdict)
    calc_q2 = solve_Gq(test_G, None, **kwdict)["phase_int"]

    allclose(calc_q1, real_q)
    allclose(calc_q2, real_q)


def test_q2G_CoQo():
    ''' Tests solve_Gq and q2G with CoQo
    '''
    kwdict = CoQo
    test_q = np.array(_test_q)
    real_G = (test_q - 0.290)/0.684
    calc_G1 = q2G(test_q, **kwdict)
    calc_G2 = solve_Gq(None, test_q, **kwdict)["slope_par"]
    allclose_Q(calc_G1, real_G*NOUNIT)
    allclose_Q(calc_G2, real_G*NOUNIT)


def test_q2G_CoQx():
    ''' Tests solve_Gq and q2G with CoQx
    '''
    kwdict = CoQx
    test_q = np.array(_test_q)
    real_G = (test_q - 0.290)/0.684
    calc_G1 = q2G(test_q, **kwdict)
    calc_G2 = solve_Gq(None, test_q, **kwdict)["slope_par"]

    allclose(calc_G1, real_G)
    allclose(calc_G2, real_G)


def test_q2G_CxQo():
    ''' Tests solve_Gq and q2G with CxQo
    '''
    kwdict = CxQo
    test_q = np.array(_test_q)
    real_G = (test_q - 0.286)/0.656
    calc_G1 = q2G(test_q, **kwdict)
    calc_G2 = solve_Gq(None, test_q, **kwdict)["slope_par"]

    allclose_Q(calc_G1, real_G*NOUNIT)
    allclose_Q(calc_G2, real_G*NOUNIT)


def test_q2G_CxQx():
    ''' Tests solve_Gq and q2G with CxQo
    '''
    kwdict = CxQx
    test_q = np.array(_test_q)
    real_G = (test_q - 0.286)/0.656
    calc_G1 = q2G(test_q, **kwdict)
    calc_G2 = solve_Gq(None, test_q, **kwdict)["slope_par"]

    allclose(calc_G1, real_G)
    allclose(calc_G2, real_G)


# Test p(p_vis), G(slope_par) and A(a_bond)
def test_pG2A_CoQo():
    ''' Tests solve_pGA and pG2A with CoQo
    '''
    kwdict = CoQo
    test_p = np.array(_test_p)

    for G in _test_G:
        real_A = (0.290 + 0.684*G) * test_p
        calc_A1 = pG2A(test_p, G, **kwdict)
        calc_A2 = solve_pAG(test_p, None, G, **kwdict)["a_bond"]
        allclose_Q(calc_A1, real_A*NOUNIT)
        allclose_Q(calc_A2, real_A*NOUNIT)


def test_pG2A_CoQx():
    ''' Tests solve_pGA and pG2A with CoQx
    '''
    kwdict = CoQx
    test_p = np.array(_test_p)

    for G in _test_G:
        real_A = (0.290 + 0.684*G) * test_p
        calc_A1 = pG2A(test_p, G, **kwdict)
        calc_A2 = solve_pAG(test_p, None, G, **kwdict)["a_bond"]
        allclose(calc_A1, real_A)
        allclose(calc_A2, real_A)


def test_pG2A_CxQo():
    ''' Tests solve_pGA and pG2A with CxQo
    '''
    kwdict = CxQo
    test_p = np.array(_test_p)

    for G in _test_G:
        real_A = (0.286 + 0.656*G) * test_p
        calc_A1 = pG2A(test_p, G, **kwdict)
        calc_A2 = solve_pAG(test_p, None, G, **kwdict)["a_bond"]
        allclose_Q(calc_A1, real_A*NOUNIT)
        allclose_Q(calc_A2, real_A*NOUNIT)


def test_pG2A_CxQx():
    ''' Tests solve_pGA and pG2A with CxQx
    '''
    kwdict = CxQx
    test_p = np.array(_test_p)

    for G in _test_G:
        real_A = (0.286 + 0.656*G) * test_p
        calc_A1 = pG2A(test_p, G, **kwdict)
        calc_A2 = solve_pAG(test_p, None, G, **kwdict)["a_bond"]
        allclose(calc_A1, real_A)
        allclose(calc_A2, real_A)


def test_AG2p_CoQo():
    ''' Tests solve_pGA and AG2p with CoQo
    '''
    kwdict = CoQo
    test_A = np.array(_test_A)

    for G in _test_G:
        real_p = test_A/(0.290 + 0.684*G)
        calc_p1 = AG2p(test_A, G, **kwdict)
        calc_p2 = solve_pAG(None, test_A, G, **kwdict)["p_vis"]
        allclose_Q(calc_p1, real_p*NOUNIT)
        allclose_Q(calc_p2, real_p*NOUNIT)


def test_AG2p_CoQx():
    ''' Tests solve_pGA and AG2p with CoQx
    '''
    kwdict = CoQx
    test_A = np.array(_test_A)

    for G in _test_G:
        real_p = test_A/(0.290 + 0.684*G)
        calc_p1 = AG2p(test_A, G, **kwdict)
        calc_p2 = solve_pAG(None, test_A, G, **kwdict)["p_vis"]
        allclose(calc_p1, real_p)
        allclose(calc_p2, real_p)


def test_AG2p_CxQo():
    ''' Tests solve_pGA and AG2p with CxQo
    '''
    kwdict = CxQo
    test_A = np.array(_test_A)

    for G in _test_G:
        real_p = test_A/(0.286 + 0.656*G)
        calc_p1 = AG2p(test_A, G, **kwdict)
        calc_p2 = solve_pAG(None, test_A, G, **kwdict)["p_vis"]
        allclose_Q(calc_p1, real_p*NOUNIT)
        allclose_Q(calc_p2, real_p*NOUNIT)


def test_AG2p_CxQx():
    ''' Tests solve_pGA and AG2p with CxQx
    '''
    kwdict = CxQx
    test_A = np.array(_test_A)

    for G in _test_G:
        real_p = test_A/(0.286 + 0.656*G)
        calc_p1 = AG2p(test_A, G, **kwdict)
        calc_p2 = solve_pAG(None, test_A, G, **kwdict)["p_vis"]
        allclose(calc_p1, real_p)
        allclose(calc_p2, real_p)


def test_pA2G_CoQo():
    ''' Tests solve_pGA and pA2G with CoQo
    '''
    kwdict = CoQo
    test_p = np.array(_test_p)

    for A in _test_A:
        real_G = (A/test_p - 0.290) / 0.684
        calc_G1 = pA2G(test_p, A, **kwdict)
        calc_G2 = solve_pAG(test_p, A, None, **kwdict)["slope_par"]
        allclose_Q(calc_G1, real_G*NOUNIT)
        allclose_Q(calc_G2, real_G*NOUNIT)


def test_pA2G_CoQx():
    ''' Tests solve_pGA and pA2G with CoQx
    '''
    kwdict = CoQx
    test_p = np.array(_test_p)

    for A in _test_A:
        real_G = (A/test_p - 0.290) / 0.684
        calc_G1 = pA2G(test_p, A, **kwdict)
        calc_G2 = solve_pAG(test_p, A, None, **kwdict)["slope_par"]
        allclose(calc_G1, real_G)
        allclose(calc_G2, real_G)


def test_pA2G_CxQo():
    ''' Tests solve_pGA and pA2G with CxQo
    '''
    kwdict = CxQo
    test_p = np.array(_test_p)

    for A in _test_A:
        real_G = (A/test_p - 0.286) / 0.656
        calc_G1 = pA2G(test_p, A, **kwdict)
        calc_G2 = solve_pAG(test_p, A, None, **kwdict)["slope_par"]
        allclose_Q(calc_G1, real_G*NOUNIT)
        allclose_Q(calc_G2, real_G*NOUNIT)


def test_pA2G_CxQx():
    ''' Tests solve_pGA and pA2G with CxQx
    '''
    kwdict = CxQx
    test_p = np.array(_test_p)

    for A in _test_A:
        real_G = (A/test_p - 0.286) / 0.656
        calc_G1 = pA2G(test_p, A, **kwdict)
        calc_G2 = solve_pAG(test_p, A, None, **kwdict)["slope_par"]
        allclose(calc_G1, real_G)
        allclose(calc_G2, real_G)


# Test p(p_vis), D(diam_eff), and H(hmag_vis)
def test_pD2H_Qo():
    ''' Tests solve_pDH and pD2H with Qo
    '''
    kwdict = Qo
    test_p = np.array(_test_p)

    for D, d0 in product(_test_D, _test_d0):
        real_H = 5 * np.log10(d0 / D * 1 / np.sqrt(test_p))
        calc_H1 = pD2H(test_p, D, d0=d0, **kwdict)
        calc_H2 = solve_pDH(test_p, D, None, d0=d0, **kwdict)["hmag_vis"]
        allclose_Q(calc_H1, real_H*NOUNIT)
        allclose_Q(calc_H2, real_H*NOUNIT)


def test_pD2H_Qx():
    ''' Tests solve_pDH and pD2H with Qx
    '''
    kwdict = Qx
    test_p = np.array(_test_p)

    for D, d0 in product(_test_D, _test_d0):
        real_H = 5 * np.log10(d0 / D * 1 / np.sqrt(test_p))
        calc_H1 = pD2H(test_p, D, d0=d0, **kwdict)
        calc_H2 = solve_pDH(test_p, D, None, d0=d0, **kwdict)["hmag_vis"]
        allclose(calc_H1, real_H)
        allclose(calc_H2, real_H)


def test_pH2D_Qo():
    ''' Tests solve_pDH and pH2D with Qo
    '''
    kwdict = Qo
    test_p = np.array(_test_p)

    for H, d0 in product(_test_H, _test_d0):
        real_D = d0 / np.sqrt(test_p)*10**(-1*H/5) * u.km
        calc_D1 = pH2D(test_p, H, d0=d0, **kwdict)
        calc_D2 = solve_pDH(test_p, None, H, d0=d0, **kwdict)["diam_eff"]
        allclose_Q(calc_D1, real_D)
        allclose_Q(calc_D2, real_D)

        d0 = d0 * u.km
        calc_D3 = pH2D(test_p, H, d0=d0, **kwdict)
        calc_D4 = solve_pDH(test_p, None, H, d0=d0, **kwdict)["diam_eff"]
        allclose_Q(calc_D3, real_D)
        allclose_Q(calc_D4, real_D)


def test_pH2D_Qx():
    ''' Tests solve_pDH and pH2D with Qx
    '''
    kwdict = Qx
    test_p = np.array(_test_p)

    for H, d0 in product(_test_H, _test_d0):
        real_D = d0 / np.sqrt(test_p)*10**(-1*H/5)
        calc_D1 = pH2D(test_p, H, d0=d0, **kwdict)
        calc_D2 = solve_pDH(test_p, None, H, d0=d0, **kwdict)["diam_eff"]
        allclose(calc_D1, real_D)
        allclose(calc_D2, real_D)

        d0 = d0 * u.km
        calc_D3 = pH2D(test_p, H, d0=d0, **kwdict)
        calc_D4 = solve_pDH(test_p, None, H, d0=d0, **kwdict)["diam_eff"]
        allclose(calc_D3, real_D)
        allclose(calc_D4, real_D)


def test_DH2p_Qo():
    ''' Tests solve_pDH and DH2p with Qo
    '''
    kwdict = Qo
    test_D = np.array(_test_D)

    for H, d0 in product(_test_H, _test_d0):
        real_p = (test_D/d0)**2*10**(-2*H/5)
        calc_p1 = DH2p(test_D, H, d0=d0, **kwdict)
        calc_p2 = solve_pDH(None, test_D, H, d0=d0, **kwdict)["p_vis"]
        allclose_Q(calc_p1, real_p*NOUNIT)
        allclose_Q(calc_p2, real_p*NOUNIT)

        d0 = d0 * u.km
        calc_p3 = DH2p(test_D, H, d0=d0, **kwdict)
        calc_p4 = solve_pDH(None, test_D, H, d0=d0, **kwdict)["p_vis"]
        allclose_Q(calc_p3, real_p*NOUNIT)
        allclose_Q(calc_p4, real_p*NOUNIT)


def test_DH2p_Qx():
    ''' Tests solve_pDH and DH2p with Qx
    '''
    kwdict = Qx
    test_D = np.array(_test_D)

    for H, d0 in product(_test_H, _test_d0):
        real_p = (test_D/d0)**2*10**(-2*H/5)
        calc_p1 = DH2p(test_D, H, d0=d0, **kwdict)
        calc_p2 = solve_pDH(None, test_D, H, d0=d0, **kwdict)["p_vis"]
        allclose_Q(calc_p1, real_p)
        allclose_Q(calc_p2, real_p)

        d0 = d0 * u.km
        calc_p3 = DH2p(test_D, H, d0=d0, **kwdict)
        calc_p4 = solve_pDH(None, test_D, H, d0=d0, **kwdict)["p_vis"]
        allclose_Q(calc_p3, real_p)
        allclose_Q(calc_p4, real_p)


def test_solve_temp_eqm_temp_eqm_Qo():
    ''' Tests solve_temp_eqm and T_eqm with Qo
    '''
    kwdict = Qo
    test_A = np.array(_test_A)

    for beam, rh, e in product(_test_beam, _test_rh, _test_eps):
        real_T = ((1 - test_A)*S1AU/(beam*SIGMA_SB*e*rh**2))**(1/4) * u.K
        calc_T1 = solve_temp_eqm(None,
                                 test_A,
                                 beam,
                                 rh,
                                 e,
                                 **kwdict
                                 )["temp_eqm"]
        calc_T2 = T_eqm(test_A, beam, rh, e, **kwdict)
        allclose_Q(calc_T1, real_T)
        allclose_Q(calc_T2, real_T)

        calc_T3 = solve_temp_eqm(None,
                                 test_A*NOUNIT,
                                 beam*NOUNIT,
                                 rh*u.au,
                                 e*NOUNIT,
                                 **kwdict
                                 )["temp_eqm"]
        calc_T4 = T_eqm(test_A*NOUNIT,
                        beam*NOUNIT,
                        rh*u.au,
                        e*NOUNIT,
                        **kwdict)
        allclose_Q(calc_T3, real_T)
        allclose_Q(calc_T4, real_T)


def test_solve_temp_eqm_temp_eqm_Qx():
    ''' Tests solve_temp_eqm and T_eqm with Qx
    '''
    kwdict = Qx
    test_A = np.array(_test_A)

    for beam, rh, e in product(_test_beam, _test_rh, _test_eps):
        real_T = ((1 - test_A)*S1AU/(beam*SIGMA_SB*e*rh**2))**(1/4)
        calc_T1 = solve_temp_eqm(None,
                                 test_A,
                                 beam,
                                 rh,
                                 e,
                                 **kwdict
                                 )["temp_eqm"]
        calc_T2 = T_eqm(test_A, beam, rh, e, **kwdict)
        allclose_Q(calc_T1, real_T)
        allclose_Q(calc_T2, real_T)

        calc_T3 = solve_temp_eqm(None,
                                 test_A*NOUNIT,
                                 beam*NOUNIT,
                                 rh*u.au,
                                 e*NOUNIT,
                                 **kwdict
                                 )["temp_eqm"]
        calc_T4 = T_eqm(test_A*NOUNIT,
                        beam*NOUNIT,
                        rh*u.au,
                        e*NOUNIT,
                        **kwdict)
        allclose(calc_T3, real_T)
        allclose(calc_T4, real_T)


def test_solve_temp_eqm_a_bond_Qo():
    ''' Tests solve_temp_eqm to get a_bond with Qo
    '''
    kwdict = Qo
    test_T = np.array(_test_T)

    for beam, rh, e in product(_test_beam, _test_rh, _test_eps):
        real_A = (1 - test_T**4*e*SIGMA_SB*beam*rh**2/S1AU) * NOUNIT
        calc_A1 = solve_temp_eqm(test_T,
                                 None,
                                 beam,
                                 rh,
                                 e,
                                 **kwdict
                                 )["a_bond"]
        calc_A2 = solve_temp_eqm(test_T*u.K,
                                 None,
                                 beam*NOUNIT,
                                 rh*u.au,
                                 e*NOUNIT,
                                 **kwdict
                                 )["a_bond"]
        allclose_Q(calc_A1, real_A)
        allclose_Q(calc_A2, real_A)


def test_solve_temp_eqm_a_bond_Qx():
    ''' Tests solve_temp_eqm to get a_bond with Qx
    '''
    kwdict = Qx
    test_T = np.array(_test_T)

    for beam, rh, e in product(_test_beam, _test_rh, _test_eps):
        real_A = (1 - test_T**4*e*SIGMA_SB*beam*rh**2/S1AU)
        calc_A1 = solve_temp_eqm(test_T,
                                 None,
                                 beam,
                                 rh,
                                 e,
                                 **kwdict
                                 )["a_bond"]
        calc_A2 = solve_temp_eqm(test_T*u.K,
                                 None,
                                 beam*NOUNIT,
                                 rh*u.au,
                                 e*NOUNIT,
                                 **kwdict
                                 )["a_bond"]
        allclose(calc_A1, real_A)
        allclose(calc_A2, real_A)


def test_solve_temp_eqm_eta_beam_Qo():
    ''' Tests solve_temp_eqm to get eta_beam with Qo
    '''
    kwdict = Qo
    test_T = np.array(_test_T)

    for A, rh, e in product(_test_A, _test_rh, _test_eps):
        real_beam = (1 - A)*S1AU/(SIGMA_SB*e*rh**2*test_T**4) * NOUNIT
        calc_beam1 = solve_temp_eqm(test_T,
                                    A,
                                    None,
                                    rh,
                                    e,
                                    **kwdict
                                    )["eta_beam"]
        calc_beam2 = solve_temp_eqm(test_T*u.K,
                                    A*NOUNIT,
                                    None,
                                    rh*u.au,
                                    e*NOUNIT,
                                    **kwdict
                                    )["eta_beam"]
        allclose_Q(calc_beam1, real_beam)
        allclose_Q(calc_beam2, real_beam)


def test_solve_temp_eqm_eta_beam_Qx():
    ''' Tests solve_temp_eqm to get eta_beam with Qx
    '''
    kwdict = Qx
    test_T = np.array(_test_T)

    for A, rh, e in product(_test_A, _test_rh, _test_eps):
        real_beam = (1 - A)*S1AU/(SIGMA_SB*e*rh**2*test_T**4)
        calc_beam1 = solve_temp_eqm(test_T,
                                    A,
                                    None,
                                    rh,
                                    e,
                                    **kwdict
                                    )["eta_beam"]
        calc_beam2 = solve_temp_eqm(test_T*u.K,
                                    A*NOUNIT,
                                    None,
                                    rh*u.au,
                                    e*NOUNIT,
                                    **kwdict
                                    )["eta_beam"]
        allclose(calc_beam1, real_beam)
        allclose(calc_beam2, real_beam)


def test_solve_temp_eqm_r_hel_Qo():
    ''' Tests solve_temp_eqm to get r_hel with Qo
    '''
    kwdict = Qo
    test_T = np.array(_test_T)

    for A, beam, e in product(_test_A, _test_beam, _test_eps):
        real_r = ((1 - A)*S1AU/(SIGMA_SB*beam*e*test_T**4))**(1/2) * u.au
        calc_r1 = solve_temp_eqm(test_T,
                                 A,
                                 beam,
                                 None,
                                 e,
                                 **kwdict
                                 )["r_hel"]
        calc_r2 = solve_temp_eqm(test_T*u.K,
                                 A*NOUNIT,
                                 beam*NOUNIT,
                                 None,
                                 e*NOUNIT,
                                 **kwdict
                                 )["r_hel"]
        allclose_Q(calc_r1, real_r)
        allclose_Q(calc_r2, real_r)


def test_solve_temp_eqm_r_hel_Qx():
    ''' Tests solve_temp_eqm to get r_hel with Qx
    '''
    kwdict = Qx
    test_T = np.array(_test_T)

    for A, beam, e in product(_test_A, _test_beam, _test_eps):
        real_r = ((1 - A)*S1AU/(SIGMA_SB*beam*e*test_T**4))**(1/2)
        calc_r1 = solve_temp_eqm(test_T,
                                 A,
                                 beam,
                                 None,
                                 e,
                                 **kwdict
                                 )["r_hel"]
        calc_r2 = solve_temp_eqm(test_T*u.K,
                                 A*NOUNIT,
                                 beam*NOUNIT,
                                 None,
                                 e*NOUNIT,
                                 **kwdict
                                 )["r_hel"]
        allclose(calc_r1, real_r)
        allclose(calc_r2, real_r)


def test_solve_temp_eqm_emissivity_Qo():
    ''' Tests solve_temp_eqm to get emissivity with Qo
    '''
    kwdict = Qo
    test_T = np.array(_test_T)

    for A, beam, rh in product(_test_A, _test_beam, _test_rh):
        real_e = (1 - A)*S1AU/(SIGMA_SB*beam*rh**2*test_T**4) * NOUNIT
        calc_e1 = solve_temp_eqm(test_T,
                                 A,
                                 beam,
                                 rh,
                                 None,
                                 **kwdict
                                 )["emissivity"]
        calc_e2 = solve_temp_eqm(test_T*u.K,
                                 A*NOUNIT,
                                 beam*NOUNIT,
                                 rh*u.au,
                                 None,
                                 **kwdict
                                 )["emissivity"]
        allclose_Q(calc_e1, real_e)
        allclose_Q(calc_e2, real_e)


def test_solve_temp_eqm_emissivity_Qx():
    ''' Tests solve_temp_eqm to get emissivity with Qx
    '''
    kwdict = Qx
    test_T = np.array(_test_T)

    for A, beam, rh in product(_test_A, _test_beam, _test_rh):
        real_e = (1 - A)*S1AU/(SIGMA_SB*beam*rh**2*test_T**4)
        calc_e1 = solve_temp_eqm(test_T,
                                 A,
                                 beam,
                                 rh,
                                 None,
                                 **kwdict
                                 )["emissivity"]
        calc_e2 = solve_temp_eqm(test_T*u.K,
                                 A*NOUNIT,
                                 beam*NOUNIT,
                                 rh*u.au,
                                 None,
                                 **kwdict
                                 )["emissivity"]
        allclose_Q(calc_e1, real_e)
        allclose_Q(calc_e2, real_e)
