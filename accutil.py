'''
This is a collection of convenience functions for my specific purpose.
Most likely you have to write your very own convenience functions or
classes for your purposes.

If you want, you may just freely use the snippets in this file.
**BUT if you do so, please make your codes be freely available for
public as these codes are!**
That's the only restriction.
'''
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.visualization import (ImageNormalize, LinearStretch,
                                   ZScaleInterval, simple_norm)
from astroquery.jplhorizons import Horizons
from matplotlib.ticker import (FormatStrFormatter, LogFormatterSciNotation,
                               LogLocator, MultipleLocator, NullFormatter)
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import UnivariateSpline as USP

import yssbtmpy as tm

__all__ = ["QPRDIR", "CHEMDICT", "EPHEMPATH",
           "PHYS_PHAE", "PHYS_PHAE_UP",
           "wrap_180180_to_000360", "wrap_000360_to_180180",
           "QprbarSunSpline", "QprbarSpline",
           "set_spl", "set_particle", "calc_traj",
           "set_phaethon", "set_phaethon_up", "set_perpmodel",
           "set_model_aspect",
           "znorm", "zimshow", "norm_imshow", "colorbaring",
           "mplticker", "linticker", "logticker", "logxticker", "logyticker"
           ]


QPRDIR = Path("./data/Qpr")
CHEMDICT = dict(oliv="Olivine", mag="Magnetite")
CHEMID = dict(ones=0, oliv=1, mag=2)
EPHEMPATH = Path("./phae_ephem.csv")

# HanusJ+2018 A&A 620 L8
PHYS_PHAE = dict(spin_ecl_lon=318*u.deg, spin_ecl_lat=-47*u.deg,
                 rot_period=3.603957*u.h, p_vis=0.12, slope_par=0.15,
                 ti=600*tm.TIU, eta_beam=1, emissivity=0.9,
                 bulk_mass_den=1670*u.kg/u.m**3,
                 diam_eff=5.1*u.km,
                 diam_eq=5.1*u.km
                 )

# HanusJ+2018 A&A 620 L8
# Diameter from TaylorPA+2019 P&SS 167 1
PHYS_PHAE_UP = dict(spin_ecl_lon=318*u.deg, spin_ecl_lat=-47*u.deg,
                    rot_period=3.603957*u.h, p_vis=0.12, slope_par=0.15,
                    ti=600*tm.TIU, eta_beam=1, emissivity=0.9,
                    bulk_mass_den=1670*u.kg/u.m**3 * (5.1/6),
                    diam_eff=6*u.km,
                    diam_eq=6*u.km
                    )
# for bulk_mass_den, Yarkovsky effect gives rho ~ 1/D


def wrap_180180_to_000360(x):
    return (x + 360) - 360*((x + 360)//360)  # [-180, +180) to [0, 360)


def wrap_000360_to_180180(x):
    return (x + 180) % 360 - 180              # [0, 360) to [-180, +180)


class QprbarSunSpline:
    def __init__(self, fpath):
        rawdata = pd.read_csv(fpath)
        rawarr = rawdata.to_numpy()[0][1:]
        radii = np.array(rawdata.columns[1:]).astype(float)
        self._spline = USP(radii, rawarr, k=3, s=0)

    def get_value(self, r_um):
        val_Qprbar = self._spline(r_um)
        return val_Qprbar.flatten()


class QprbarSpline:
    def __init__(self, fpath):
        rawdata = pd.read_csv(fpath)
        rawarr = rawdata.to_numpy()[:, 1:]
        radii = np.array(rawdata.columns[1:]).astype(float)
        self._spline = RectBivariateSpline(rawdata["T"], radii, rawarr,
                                           kx=3, ky=3, s=0)

    def get_value(self, T_K, r_um):
        val_Qprbar = self._spline(T_K, r_um)
        return val_Qprbar.flatten()


def set_spl(fpath=EPHEMPATH):
    if not Path(fpath).exists():
        epoch_ref = Time(2456049.8189178086, format='jd')
        # perihelion: 2012-05-02T07:39:14.499
        epochs_peri = epoch_ref + np.arange(-5, +5, 0.1) * u.day
        epochs_long = dict(start=(epoch_ref - 300.01*u.day).isot,
                           stop=(epoch_ref + 300.01*u.day).isot,
                           step='1d')
        obj_peri = Horizons(id=3200, epochs=epochs_peri.jd)
        obj_long = Horizons(id=3200, epochs=epochs_long)
        eph_peri = obj_peri.ephemerides()
        eph_long = obj_long.ephemerides()

        eph_all = vstack([eph_peri, eph_long])
        eph_all.sort(keys="true_anom")

        pos_ecl = tm.lonlat2cart(lon=eph_all["EclLon"], lat=eph_all["EclLat"])
        spin_ecl = tm.lonlat2cart(lon=PHYS_PHAE["spin_ecl_lon"],
                                  lat=PHYS_PHAE["spin_ecl_lat"])
        _theta_asp = np.rad2deg(np.arccos(np.inner(pos_ecl.T, spin_ecl)))
        eph_all["theta_asp"] = 180 - _theta_asp
        eph_all.write(fpath)

    eph_all = Table.read(fpath)

    # as functions of true anomaly:
    f = eph_all["true_anom"]
    # NOTE: f is 0~360 notation from JPL HORIZONS
    ks = dict(k=3, s=0)
    spl_rh = USP(f, eph_all["r"], **ks)
    spl_asp = USP(f, eph_all["theta_asp"], **ks)
    spl_ro = USP(f, eph_all["delta"], **ks)
    spl_hlon = USP(f, eph_all["EclLon"], **ks)
    spl_hlat = USP(f, eph_all["EclLat"], **ks)
    spl_olon = USP(f, eph_all["ObsEclLon"], **ks)
    spl_olat = USP(f, eph_all["ObsEclLat"], **ks)
    spl_alpha = USP(f, eph_all["alpha"], **ks)

    return dict(rh=spl_rh, theta_asp=spl_asp, ro=spl_ro,
                hlon=spl_hlon, hlat=spl_hlat,
                olon=spl_olon, olat=spl_olat,
                alpha=spl_alpha)


def set_phaethon(true_anom=0, ti=600, nlon=360, nlat=180,
                 fpath=EPHEMPATH):
    # Just in case it is in -180~+180 notation:
    true_anom = wrap_180180_to_000360(true_anom)
    spl = set_spl()
    sb = tm.SmallBody()
    sb.id = 3200
    sb.set_ecl(r_hel=spl["rh"](true_anom),
               hel_ecl_lon=spl["hlon"](true_anom),
               hel_ecl_lat=spl["hlat"](true_anom),
               r_obs=spl["ro"](true_anom),  # dummy....
               obs_ecl_lon=spl["olon"](true_anom),
               obs_ecl_lat=spl["olat"](true_anom),
               alpha=spl["alpha"](true_anom)
               )

    sb.set_spin(spin_ecl_lon=PHYS_PHAE["spin_ecl_lon"],
                spin_ecl_lat=PHYS_PHAE["spin_ecl_lat"],
                rot_period=PHYS_PHAE["rot_period"])
    sb.set_optical(slope_par=PHYS_PHAE["slope_par"],
                   diam_eff=PHYS_PHAE["diam_eff"],
                   p_vis=PHYS_PHAE["p_vis"])
    sb.set_mass(diam_eff=PHYS_PHAE["diam_eff"],
                bulk_mass_den=PHYS_PHAE["bulk_mass_den"])
    sb.set_thermal(ti=ti, emissivity=PHYS_PHAE["emissivity"])
    sb.set_tpm(nlon=nlon, nlat=nlat, Zmax=10, nZ=50)
    return sb


def set_phaethon_up(true_anom=0, ti=600, nlon=360, nlat=180,
                    fpath=EPHEMPATH):
    spl = set_spl()
    sb = tm.SmallBody()
    sb.id = 3200
    sb.set_ecl(r_hel=spl["rh"](true_anom),
               hel_ecl_lon=spl["hlon"](true_anom),
               hel_ecl_lat=spl["hlat"](true_anom),
               r_obs=spl["ro"](true_anom),  # dummy....
               obs_ecl_lon=spl["olon"](true_anom),
               obs_ecl_lat=spl["olat"](true_anom),
               alpha=spl["alpha"](true_anom)
               )

    sb.set_spin(spin_ecl_lon=PHYS_PHAE_UP["spin_ecl_lon"],
                spin_ecl_lat=PHYS_PHAE_UP["spin_ecl_lat"],
                rot_period=PHYS_PHAE_UP["rot_period"])
    sb.set_optical(slope_par=PHYS_PHAE_UP["slope_par"],
                   diam_eff=PHYS_PHAE_UP["diam_eff"],
                   p_vis=PHYS_PHAE_UP["p_vis"])
    sb.set_mass(diam_eff=PHYS_PHAE_UP["diam_eff"],
                bulk_mass_den=PHYS_PHAE_UP["bulk_mass_den"])
    sb.set_thermal(ti=ti, emissivity=PHYS_PHAE_UP["emissivity"])
    sb.set_tpm(nlon=nlon, nlat=nlat, Zmax=10, nZ=50)
    return sb


def set_perpmodel(diam_eff, rot_period, r_hel=0.2,
                  a_bond=0.1, ti=200, bulk_mass_den=2000, emissivity=0.90,
                  nlon=360, nlat=180):
    dummies = dict(r_obs=1, obs_ecl_lon=0, obs_ecl_lat=0, alpha=0)

    sb = tm.SmallBody()
    sb.set_ecl(r_hel=r_hel, hel_ecl_lon=0, hel_ecl_lat=0, **dummies)
    sb.set_spin(spin_ecl_lon=0, spin_ecl_lat=90, rot_period=rot_period)
    sb.set_optical(slope_par=0, a_bond=a_bond, diam_eff=diam_eff,
                   p_vis=tm.AG2p(a_bond, 0))
    sb.set_mass(diam_eff=diam_eff, bulk_mass_den=bulk_mass_den)
    sb.set_thermal(ti=ti, emissivity=emissivity)
    sb.set_tpm(nlon=nlon, nlat=nlat, Zmax=10, nZ=50)
    return sb


def set_model_aspect(diam_eff, rot_period, aspect_deg=90, r_hel=0.2,
                     a_bond=0.1, ti=200, bulk_mass_den=2000, emissivity=0.90,
                     nlon=360, nlat=180):
    dummies = dict(r_obs=1, obs_ecl_lon=0, obs_ecl_lat=0, alpha=0)

    sb = tm.SmallBody()
    sb.set_ecl(r_hel=r_hel, hel_ecl_lon=0, hel_ecl_lat=0, **dummies)
    sb.set_spin(spin_ecl_lon=0, spin_ecl_lat=aspect_deg, rot_period=rot_period)
    sb.set_optical(slope_par=0, a_bond=a_bond, diam_eff=diam_eff,
                   p_vis=tm.AG2p(a_bond, 0))
    sb.set_mass(diam_eff=diam_eff, bulk_mass_den=bulk_mass_den)
    sb.set_thermal(ti=ti, emissivity=emissivity)
    sb.set_tpm(nlon=nlon, nlat=nlat, Zmax=10, nZ=50)
    return sb


def set_particle(smallbody, radius_um, chem, init_th, init_ph,
                 Qprbar_sun=None, Qprbar_ther=None,
                 init_height=1*u.cm, mass_den=3000*u.kg/u.m**3,
                 r0_radius=0.01, vec_vel_init=None):
    # Splining Qprbar is the most time-consuming part (~ 20ms)
    # compared to all others (< 1ms), so better to give a priori if possible.
    if Qprbar_sun is None:
        Qprbar_sun = QprbarSunSpline(QPRDIR/f"Qprbar_sun_{chem}.csv")
    if Qprbar_ther is None:
        Qprbar_ther = QprbarSpline(QPRDIR/f"Qprbar_{chem}.csv")

    particle = tm.MovingParticle(smallbody=smallbody,
                                 radius=radius_um*u.um,
                                 mass_den=mass_den,
                                 r0_radius=r0_radius)

    particle.set_func_Qprbar(func_Qprbar=Qprbar_ther.get_value,
                             func_Qprbar_sun=Qprbar_sun.get_value)
    particle.set_initial_pos(init_th, init_ph,
                             height=init_height, vec_vel_init=vec_vel_init)
    return particle


def calc_traj(chem, radius, sb, init_th, init_ph, max_h_step, nstep=None,
              min_height=0.1*u.cm, init_height=1*u.cm, r0_radius=0.01,
              vec_vel_init=None,
              mass_den=3000*u.kg/u.m**3, dt=0.1, return_halt_code=False):
    particle = set_particle(sb, radius, chem, init_th, init_ph,
                            init_height=init_height, mass_den=mass_den,
                            r0_radius=r0_radius,
                            vec_vel_init=vec_vel_init)
    sb_GM = tm.GG * sb.mass.to(u.kg).value

    max_h = 1*max_h_step
    min_h = 1*min_height
    halt_code_str_2 = None
    if max_h_step is not None:
        while True:
            PROP_KW = dict(dt=dt, verbose=False, nstep=nstep,
                           min_height=min_h, max_height=max_h)
            particle.propagate(**PROP_KW)
            if particle.halt_code_str == 'max_height':
                r = particle.trace_pos_sph[-1][0]
                pos_last = particle.trace_pos_xyz[-1]
                # phi_last = particle.trace_pos_sph[-1][2]
                # phi_init = particle.trace_pos_sph[0][2]
                vel_last = particle.trace_vel_xyz[-1]
                h_last = particle.trace_height[-1]
                vel_esc = np.sqrt(2*sb_GM/r)

                t_last = particle.trace_time[-1]

                # vel_x > vel_esc && pos_x > 0 (night hemisphere)
                if (vel_last[0] > vel_esc) and (pos_last[0] > 0):
                    halt_code_str_2 = 'escape'
                    # particle.halt_code_str = 'escape'
                    break
                elif t_last > dt*nstep:
                    halt_code_str_2 = 'max_time'
                    # particle.halt_code = 4
                    # particle.halt_code_str = 'max_time'
                    break
                # elif abs(phi_init - phi_last) > 90:
                #     halt_code_str_2 = 'oscillation'
                #     break
                else:
                    max_h += max_h_step
                    min_h = max(min_height.to(u.m).value,
                                h_last - max_h_step.to(u.m).value)
            elif particle.halt_code_str == 'min_height':
                h_last = particle.trace_height[-1]
                if h_last > min_height.to(u.m).value:
                    halt_code_str_2 = 'oscillation'
                else:
                    halt_code_str_2 = 'min_height'
                break
            else:
                halt_code_str_2 = 'none'
                break
    else:
        PROP_KW = dict(dt=dt, verbose=False, nstep=nstep,
                       min_height=min_height, max_height=None)
        particle.propagate(**PROP_KW)

    particle.wrapup()
    if return_halt_code:
        return particle, halt_code_str_2
    return particle


def ax_tick(ax, x_vals=None, x_show=None, y_vals=None, y_show=None):
    # if (x_vals is None) ^ (x_show is None):
    #     raise ValueError("All or none of x_vals and x_show should be given.")
    # if (y_vals is None) ^ (y_show is None):
    #     raise ValueError("All or none of y_vals and y_show should be given.")

    if x_vals is not None:
        x_vals = np.array(x_vals)
        if x_show is None:
            x_ticks = x_vals.copy()
            x_show = x_vals.copy()
        else:
            x_ticks = np.array([np.where(x_vals == v)[0][0] for v in x_show])

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_show)

    if y_vals is not None:
        y_vals = np.array(y_vals)
        if y_show is None:
            y_ticks = y_vals.copy()
            y_show = y_vals.copy()
        else:
            y_ticks = np.array([np.where(y_vals == v)[0][0] for v in y_show])

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_show)

    return ax


########################################################################
# Below are from ysfitsutilpy.visutil module from version 2019-09-14,
# commit number bc3e0b7037a1a4a766972abe053acc22c7e4b016
########################################################################


def znorm(image, stretch=LinearStretch(), **kwargs):
    return ImageNormalize(image,
                          interval=ZScaleInterval(**kwargs),
                          stretch=stretch)


def zimshow(ax, image, stretch=LinearStretch(), cmap=None, **kwargs):
    im = ax.imshow(image,
                   norm=znorm(image, stretch=stretch, **kwargs),
                   origin='lower',
                   cmap=cmap)
    return im


def norm_imshow(ax, data, stretch='linear', power=1.0, asinh_a=0.1,
                min_cut=None, max_cut=None, min_percent=None, max_percent=None,
                percent=None, clip=True, log_a=1000, **kwargs):
    """ Do normalization and do imshow
    """
    norm = simple_norm(data, stretch=stretch, power=power, asinh_a=asinh_a,
                       min_cut=min_cut, max_cut=max_cut,
                       min_percent=min_percent, max_percent=max_percent,
                       percent=percent, clip=clip, log_a=log_a)
    im = ax.imshow(data, norm=norm, origin='lower', **kwargs)
    return im


def colorbaring(fig, ax, im, fmt="%.0f", orientation='horizontal',
                formatter=FormatStrFormatter, **kwargs):
    cb = fig.colorbar(im, ax=ax, orientation=orientation,
                      format=FormatStrFormatter(fmt), **kwargs)

    return cb


def mplticker(ax_list,
              xmajlocators=None, xminlocators=None,
              ymajlocators=None, yminlocators=None,
              xmajformatters=None, xminformatters=None,
              ymajformatters=None, yminformatters=None,
              xmajgrids=True, xmingrids=True,
              ymajgrids=True, ymingrids=True,
              xmajlockws=None, xminlockws=None,
              ymajlockws=None, yminlockws=None,
              xmajfmtkws=None, xminfmtkws=None,
              ymajfmtkws=None, yminfmtkws=None,
              xmajgridkws=dict(ls='-', alpha=0.5),
              xmingridkws=dict(ls=':', alpha=0.5),
              ymajgridkws=dict(ls='-', alpha=0.5),
              ymingridkws=dict(ls=':', alpha=0.5)):
    ''' Set tickers of Axes objects.
    Note
    ----
    Notation of arguments is <axis><which><name>. <axis> can be ``x`` or
    ``y``, and <which> can be ``major`` or ``minor``.
    For example, ``xmajlocators`` is the Locator object for x-axis
    major.  ``kw`` means the keyword arguments that will be passed to
    locator, formatter, or grid.
    If a single object is given for locators, formatters, grid, or kw
    arguments, it will basically be copied by the number of Axes objects
    and applied identically through all the Axes.

    Parameters
    ----------
    ax_list : Axes or 1-d array-like of such
        The Axes object(s).

    locators : Locator, None, list of such, optional
        The Locators used for the ticker. Must be a single Locator
        object or a list of such with the identical length of
        ``ax_list``.
        If ``None``, the default locator is not touched.

    formatters : Formatter, None, False, array-like of such, optional
        The Formatters used for the ticker. Must be a single Formatter
        object or an array-like of such with the identical length of
        ``ax_list``.
        If ``None``, the default formatter is not touched.
        If ``False``, the labels are not shown (by using the trick
        ``FormatStrFormatter(fmt="")``).

    grids : bool, array-like of such, optinoal.
        Whether to draw the grid lines. Must be a single bool object or
        an array-like of such with the identical length of ``ax_list``.

    lockws : dict, array-like of such, array-like, optional
        The keyword arguments that will be passed to the ``locators``.
        If it's an array-like but elements are not dict, it is
        interpreted as ``*args`` passed to locators.
        If it is (or contains) dict, it must be a single dict object or
        an array-like object of such with the identical length of
        ``ax_list``.
        Set as empty dict (``{}``) if you don't want to change the
        default arguments.

    fmtkws : dict, str, list of such, optional
        The keyword arguments that will be passed to the ``formatters``.
        If it's an array-like but elements are not dict, it is
        interpreted as ``*args`` passed to formatters.
        If it is (or contains) dict, it must be a single dict object or
        an array-like object of such with the identical length of
        ``ax_list``.
        Set as empty dict (``{}``) if you don't want to change the
        default arguments.

    gridkw : dict, list of such, optional
        The keyword arguments that will be passed to the grid. Must be a
        single dict object or a list of such with the identical length
        of ``ax_list``.
        Set as empty dict (``{}``) if you don't want to change the
        default arguments.

    '''
    def _check(obj, name, n):
        arr = np.atleast_1d(obj)
        n_arr = arr.shape[0]
        if n_arr not in [1, n]:
            raise ValueError(f"{name} must be a single object or a 1-d array"
                             + f" with the same length as ax_list ({n}).")
        else:
            newarr = arr.tolist() * (n//n_arr)

        return newarr

    def _setter(setter, Setter, kw):
        # don't do anything if obj (Locator or Formatter) is None:
        if (Setter is not None) and (kw is not None):

            # matplotlib is so poor in log plotting....
            if (Setter == LogLocator) and ("numticks" not in kw):
                kw["numticks"] = 50

            if isinstance(kw, dict):
                setter(Setter(**kw))
            else:  # interpret as ``*args``
                setter(Setter(*(np.atleast_1d(kw).tolist())))
            # except:
            #     raise ValueError("Error occured for Setter={} with input {}"
            #                      .format(Setter, kw))

    _ax_list = np.atleast_1d(ax_list)
    if _ax_list.ndim > 1:
        raise ValueError("ax_list must be at most 1-d.")
    n_axis = _ax_list.shape[0]

    _xmajlocators = _check(xmajlocators, "xmajlocators", n_axis)
    _xminlocators = _check(xminlocators, "xminlocators", n_axis)
    _ymajlocators = _check(ymajlocators, "ymajlocators", n_axis)
    _yminlocators = _check(yminlocators, "yminlocators", n_axis)

    _xmajformatters = _check(xmajformatters, "xmajformatters ", n_axis)
    _xminformatters = _check(xminformatters, "xminformatters ", n_axis)
    _ymajformatters = _check(ymajformatters, "ymajformatters", n_axis)
    _yminformatters = _check(yminformatters, "yminformatters", n_axis)

    _xmajlockws = _check(xmajlockws, "xmajlockws", n_axis)
    _xminlockws = _check(xminlockws, "xminlockws", n_axis)
    _ymajlockws = _check(ymajlockws, "ymajlockws", n_axis)
    _yminlockws = _check(yminlockws, "yminlockws", n_axis)

    _xmajfmtkws = _check(xmajfmtkws, "xmajfmtkws", n_axis)
    _xminfmtkws = _check(xminfmtkws, "xminfmtkws", n_axis)
    _ymajfmtkws = _check(ymajfmtkws, "ymajfmtkws", n_axis)
    _yminfmtkws = _check(yminfmtkws, "yminfmtkws", n_axis)

    _xmajgrids = _check(xmajgrids, "xmajgrids", n_axis)
    _xmingrids = _check(xmingrids, "xmingrids", n_axis)
    _ymajgrids = _check(ymajgrids, "ymajgrids", n_axis)
    _ymingrids = _check(ymingrids, "ymingrids", n_axis)

    _xmajgridkws = _check(xmajgridkws, "xmajgridkws", n_axis)
    _xmingridkws = _check(xmingridkws, "xmingridkws", n_axis)
    _ymajgridkws = _check(ymajgridkws, "ymajgridkws", n_axis)
    _ymingridkws = _check(ymingridkws, "ymingridkws", n_axis)

    for i, aa in enumerate(_ax_list):
        _xmajlocator = _xmajlocators[i]
        _xminlocator = _xminlocators[i]
        _ymajlocator = _ymajlocators[i]
        _yminlocator = _yminlocators[i]

        _xmajformatter = _xmajformatters[i]
        _xminformatter = _xminformatters[i]
        _ymajformatter = _ymajformatters[i]
        _yminformatter = _yminformatters[i]

        _xmajgrid = _xmajgrids[i]
        _xmingrid = _xmingrids[i]
        _ymajgrid = _ymajgrids[i]
        _ymingrid = _ymingrids[i]

        _xmajlockw = _xmajlockws[i]
        _xminlockw = _xminlockws[i]
        _ymajlockw = _ymajlockws[i]
        _yminlockw = _yminlockws[i]

        _xmajfmtkw = _xmajfmtkws[i]
        _xminfmtkw = _xminfmtkws[i]
        _ymajfmtkw = _ymajfmtkws[i]
        _yminfmtkw = _yminfmtkws[i]

        _xmajgridkw = _xmajgridkws[i]
        _xmingridkw = _xmingridkws[i]
        _ymajgridkw = _ymajgridkws[i]
        _ymingridkw = _ymingridkws[i]

        _setter(aa.xaxis.set_major_locator, _xmajlocator, _xmajlockw)
        _setter(aa.xaxis.set_minor_locator, _xminlocator, _xminlockw)
        _setter(aa.yaxis.set_major_locator, _ymajlocator, _ymajlockw)
        _setter(aa.yaxis.set_minor_locator, _yminlocator, _yminlockw)
        _setter(aa.xaxis.set_major_formatter, _xmajformatter, _xmajfmtkw)
        _setter(aa.xaxis.set_minor_formatter, _xminformatter, _xminfmtkw)
        _setter(aa.yaxis.set_major_formatter, _ymajformatter, _ymajfmtkw)
        _setter(aa.yaxis.set_minor_formatter, _yminformatter, _yminfmtkw)

        # Strangely, using ``b=_xmingrid`` does not work if it is
        # False... I had to do it manually like this... OMG matplotlib..
        if _xmajgrid:
            aa.grid(axis='x', which='major', **_xmajgridkw)
        if _xmingrid:
            aa.grid(axis='x', which='minor', **_xmingridkw)
        if _ymajgrid:
            aa.grid(axis='y', which='major', **_ymajgridkw)
        if _ymingrid:
            aa.grid(axis='y', which='minor', **_ymingridkw)


def linticker(ax_list,
              xmajlocators=MultipleLocator, xminlocators=MultipleLocator,
              ymajlocators=MultipleLocator, yminlocators=MultipleLocator,
              xmajformatters=FormatStrFormatter,
              xminformatters=NullFormatter,
              ymajformatters=FormatStrFormatter,
              yminformatters=NullFormatter,
              xmajgrids=True, xmingrids=True,
              ymajgrids=True, ymingrids=True,
              xmajlockws=None, xminlockws=None,
              ymajlockws=None, yminlockws=None,
              xmajfmtkws=None, xminfmtkws={},
              ymajfmtkws=None, yminfmtkws={},
              xmajgridkws=dict(ls='-', alpha=0.5),
              xmingridkws=dict(ls=':', alpha=0.5),
              ymajgridkws=dict(ls='-', alpha=0.5),
              ymingridkws=dict(ls=':', alpha=0.5)):
    mplticker(**locals())


def logticker(ax_list,
              xmajlocators=LogLocator, xminlocators=LogLocator,
              ymajlocators=LogLocator, yminlocators=LogLocator,
              xmajformatters=LogFormatterSciNotation,
              xminformatters=NullFormatter,
              ymajformatters=LogFormatterSciNotation,
              yminformatters=NullFormatter,
              xmajgrids=True, xmingrids=True,
              ymajgrids=True, ymingrids=True,
              xmajlockws=None, xminlockws=None,
              ymajlockws=None, yminlockws=None,
              xmajfmtkws=None, xminfmtkws={},
              ymajfmtkws=None, yminfmtkws={},
              xmajgridkws=dict(ls='-', alpha=0.5),
              xmingridkws=dict(ls=':', alpha=0.5),
              ymajgridkws=dict(ls='-', alpha=0.5),
              ymingridkws=dict(ls=':', alpha=0.5)):
    mplticker(**locals())


def logxticker(ax_list,
               xmajlocators=LogLocator, xminlocators=LogLocator,
               ymajlocators=MultipleLocator, yminlocators=MultipleLocator,
               xmajformatters=LogFormatterSciNotation,
               xminformatters=NullFormatter,
               ymajformatters=FormatStrFormatter,
               yminformatters=NullFormatter,
               xmajgrids=True, xmingrids=True,
               ymajgrids=True, ymingrids=True,
               xmajlockws=None, xminlockws=None,
               ymajlockws=None, yminlockws=None,
               xmajfmtkws=None, xminfmtkws={},
               ymajfmtkws=None, yminfmtkws={},
               xmajgridkws=dict(ls='-', alpha=0.5),
               xmingridkws=dict(ls=':', alpha=0.5),
               ymajgridkws=dict(ls='-', alpha=0.5),
               ymingridkws=dict(ls=':', alpha=0.5)):
    mplticker(**locals())


def logyticker(ax_list,
               xmajlocators=MultipleLocator, xminlocators=MultipleLocator,
               ymajlocators=LogLocator, yminlocators=LogLocator,
               xmajformatters=FormatStrFormatter,
               xminformatters=NullFormatter,
               ymajformatters=LogFormatterSciNotation,
               yminformatters=NullFormatter,
               xmajgrids=True, xmingrids=True,
               ymajgrids=True, ymingrids=True,
               xmajlockws=None, xminlockws=None,
               ymajlockws=None, yminlockws=None,
               xmajfmtkws=None, xminfmtkws={},
               ymajfmtkws=None, yminfmtkws={},
               xmajgridkws=dict(ls='-', alpha=0.5),
               xmingridkws=dict(ls=':', alpha=0.5),
               ymajgridkws=dict(ls='-', alpha=0.5),
               ymingridkws=dict(ls=':', alpha=0.5)):
    mplticker(**locals())
