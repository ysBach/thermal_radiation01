import datetime
from warnings import warn

import numpy as np
from astropy import units as u
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
from astropy.coordinates import SkyCoord, HeliocentricMeanEcliptic

from .constants import GG_Q, NOUNIT, PI, R2D, TIU
from .relations import (solve_Gq, solve_pAG, solve_pDH, solve_rmrho,
                        solve_temp_eqm, solve_thermal_par)
from .util import (add_hdr, calc_mu_suns, change_to_quantity, setup_uarr_tpm,
                   lonlat2cart)

__all__ = ["SmallBody"]

# TODO: Maybe we can inherit the Phys class from sbpy to this, so that
#   the physical data (hmag_vis, Prot, etc) is imported from, e.g., SBDB
#   by default.


class SmallBody():
    ''' Spherical Small Body class
    Specified by physical parameters that are the body's
    characteristic, not time-variable ones (e.g., ephemerides).

    Example
    -------
    >>> test = tm.SmallBody()
    >>> test.set_ecl(
    >>>     r_hel=1.5, r_obs=0.5, alpha=0,
    >>>     hel_ecl_lon=0, hel_ecl_lat=0,
    >>>     obs_ecl_lon=0, obs_ecl_lat=0
    >>> )
    >>> test.set_spin(
    >>>     spin_ecl_lon=10, spin_ecl_lat=-90, rot_period=1
    >>> )
    >>> test.aspect_ang, test.aspect_ang_obs
    Test by varying spin vector. In all cases tested below, calculation
    matched with desired values:
        ----------input-----------  ---------desired----------
        spin_ecl_lon  spin_ecl_lat  aspect_ang  aspect_ang_obs
        0             0             180         180
        +- 30, 330    0             150         150
        0             +-30          150         150
        any value     +-90          90          90

    >>> test = tm.SmallBody()
    >>> test.set_ecl(
    >>>     r_hel=1.414, r_obs=1, alpha=45,
    >>>     hel_ecl_lon=45, hel_ecl_lat=0,
    >>>     obs_ecl_lon=90, obs_ecl_lat=0
    >>> )
    >>> test.set_spin(
    >>>     spin_ecl_lon=330, spin_ecl_lat=0, rot_period=1
    >>> )
    >>> test.aspect_ang, test.aspect_ang_obs
    Test by varying spin vector. In all cases tested below, calculation
    matched with desired values:
        ----------input-----------  ---------desired----------
        spin_ecl_lon  spin_ecl_lat  aspect_ang  aspect_ang_obs
        0             0             135         90
        + 30          0             165         120
        - 30, 330     0             105         60
        0             +-30          127.76      90
        any value     +-90          90          90

    '''

    # TODO: What if a user input values with Quantity?
    #   Removing the units is not what is desired...
    def __init__(self):
        self.id = None
        # self.use_quantity = use_quantity

        # Spin related
        self.spin_ecl_lon = None
        self.spin_ecl_lat = None
        self.spin_vec = np.array([None, None, None])
        self.rot_period = None
        self.rot_omega = None

        # Epemerides related
        self.obs_ecl_lon = None
        self.obs_ecl_lat = None
        self.hel_ecl_lon = None
        self.hel_ecl_lat = None
        self.r_hel = None
        self.r_hel_vec = np.array([None, None, None])
        self.r_obs = None
        self.r_obs_vec = np.array([None, None, None])
        self.phase_ang = None

        self.aspect_ang = None

        # Optical related
        self.hmag_vis = None
        self.slope_par = None
        self.phase_int = None
        self.diam_eff = None
        self.radi_eff = None
        self.p_vis = None
        self.a_bond = None

        # physics related
        self.mass = None
        self.bulk_mass_den = None
        # self.diam_equat = None
        self.acc_grav_equator = None
        self.v_esc_equator = None

        # TPM physics related
        self.ti = None
        self.thermal_par = None
        self.eta_beam = None
        self.emissivity = None
        self.temp_eqm = None
        self.temp_eqm_1au = None

        # TPM code related
        self.nlon = None
        self.nlat = None
        self.dlon = None
        self.dlat = None
        self.Zmax = None
        self.nZ = None
        self.dZ = None

        self.mu_suns = None
        self.tempfull = None
        self.tempsurf = None
        self.tempunit = None

    def _set_aspect_angle(self):
        if ((self.r_hel_vec != np.array([None, None, None]))
                and (self.r_obs_vec != np.array([None, None, None]))
                and (self.aspect_ang is None)):
            r_hel_hat = self.r_hel_vec/self.r_hel
            r_obs_hat = self.r_obs_vec/self.r_obs
            # r_obs_hat = lonlat2cart(self.obs_ecl_helio.lon,
            #                         self.obs_ecl_helio.lat)

            asp1 = np.rad2deg(np.arccos(np.inner(-1*r_hel_hat, self.spin_vec)))
            asp2 = np.rad2deg(np.arccos(np.inner(-1*r_obs_hat, self.spin_vec)))

            toQ = dict(to_value=False)
            self.aspect_ang = change_to_quantity(asp1, u.deg, **toQ)
            # aux_cos_sun = np.inner(r_hel_hat, self.spin_vec)
            # self.aspect_ang = (180-np.rad2deg(np.arccos(aux_cos_sun)))*u.deg
            self.aspect_ang_obs = change_to_quantity(asp2, u.deg, **toQ)
            # aux_cos_obs = np.inner(r_obs_hat, self.spin_vec)
            # aspect_ang_obs = (180-np.rad2deg(np.arccos(aux_cos_obs)))*u.deg
            # ([(-r_obs) x (-r_sun)] \cdot spin) has opposite sign of
            # dphi (the phi difference between the subsolar and
            # subobserver points) :
            sign = -1*np.sign(np.inner(np.cross(r_obs_hat, r_hel_hat),
                                       self.spin_vec))
            cc = np.cos(self.aspect_ang) * np.cos(self.aspect_ang_obs)
            ss = np.sin(self.aspect_ang) * np.sin(self.aspect_ang_obs)
            _arg = ((cc - np.cos(self.phase_ang))/ss).value
            if _arg < -1:
                _arg += 1.e-10
            elif _arg > +1:
                _arg -= 1.e-10
            dphi = change_to_quantity(np.arccos(_arg), u.deg, **toQ)
            phi_obs = (180*u.deg + sign*dphi).to(u.deg)
            if np.isnan(phi_obs):
                raise ValueError("Oops T___T")
            self.pos_sub_sol = (self.aspect_ang, 180*u.deg)
            self.pos_sub_obs = (self.aspect_ang_obs, phi_obs)

    def set_ecl(self, r_hel, hel_ecl_lon, hel_ecl_lat,
                r_obs, obs_ecl_lon, obs_ecl_lat, alpha,
                ephem_equinox='J2000.0', transform_equinox='J2000.0'):
        '''
        Note
        ----
        The ``ObsEcLon`` and ``ObsEcLat`` from JPL HORIZONS are in the
        equinox of the observing time, not J2000.0. Although this
        difference will not give large uncertainty, this may be
        uncomfortable for some application purposes. In this case, give
        the ephemerides date (such as
        ``ephem_equinox=Time(eph['datetime_jd'], format='jd')``) and the
        equinox of ``hEcl-Lon`` and ``hEcl-Lat`` is calculated (as of
        2019, it is J2000.0, so ``transform_equinox="J2000.0"``).
        '''
        toQ = dict(to_value=False)
        self.r_hel = change_to_quantity(r_hel, u.au, **toQ)
        self.r_obs = change_to_quantity(r_obs, u.au, **toQ)
        self.hel_ecl_lon = change_to_quantity(hel_ecl_lon, u.deg, **toQ)
        self.hel_ecl_lat = change_to_quantity(hel_ecl_lat, u.deg, **toQ)
        # The obs_ecl values from HORIZONS are
        self.obs_ecl_lon = change_to_quantity(obs_ecl_lon, u.deg, **toQ)
        self.obs_ecl_lat = change_to_quantity(obs_ecl_lat, u.deg, **toQ)
        self.phase_ang = change_to_quantity(alpha, u.deg, **toQ)

        # helecl_ref = HeliocentricMeanEcliptic(equinox=transform_equinox)
        # obsecl_geo = SkyCoord(
        #     self.obs_ecl_lon,
        #     self.obs_ecl_lat,
        #     self.r_obs,
        #     equinox=ephem_equinox
        # )
        # self.obs_ecl_helio = obsecl_geo.transform_to(helecl_ref)

        try:
            vec = lonlat2cart(lon=self.hel_ecl_lon,
                              lat=self.hel_ecl_lat,
                              r=self.r_hel.value)
            self.r_hel_vec = vec*u.au
        except TypeError:
            self.r_hel_vec = np.array([None, None, None])

        try:
            vec = lonlat2cart(lon=self.obs_ecl_lon,
                              lat=self.obs_ecl_lat,
                              r=self.r_obs.value)
            self.r_obs_vec = vec*u.au
        except TypeError:
            self.r_obs_vec = np.array([None, None, None])

        try:
            self._set_aspect_angle()

        except TypeError:
            pass

    def set_spin(self, spin_ecl_lon, spin_ecl_lat, rot_period=None):
        toQ = dict(to_value=False)
        self.spin_ecl_lon = change_to_quantity(spin_ecl_lon, u.deg, **toQ)
        self.spin_ecl_lat = change_to_quantity(spin_ecl_lat, u.deg, **toQ)
        self.rot_period = change_to_quantity(rot_period, u.s, **toQ)

        try:
            vec = lonlat2cart(lon=self.spin_ecl_lon,
                              lat=self.spin_ecl_lat, r=1)
            self.spin_vec = vec  # unit vector
        except TypeError:
            self.spin_vec = np.array([None, None, None])

        try:
            rot_omega = 2*PI/(self.rot_period)
            self.rot_omega = rot_omega  # .to('rad/s')
        except TypeError:
            self.rot_omega = None

        try:
            self._set_aspect_angle()
        except TypeError:
            pass

    def set_mass(self, diam_eff=None, mass=None, bulk_mass_den=None):
        ps = solve_rmrho(radius=diam_eff/2, mass=mass, mass_den=bulk_mass_den)
        ps["diam_eff"] = 2*ps["radius"]
        for p in ["diam_eff", "mass", "bulk_mass_den", "acc_grav_equator"]:
            if getattr(self, p) is not None:
                try:
                    u.allclose(getattr(self, p), ps[p])
                except AssertionError:
                    warn(f"self.{p} is not None ({getattr(self, p)}), "
                         + f"and will be overridden by {ps[p]}.")

        self.diam_eff = ps["diam_eff"]
        self.radi_eff = self.diam_eff/2
        self.mass = ps["mass"]
        self.bulk_mass_den = ps["mass_den"]
        self.acc_grav_equator = GG_Q*self.mass/(self.radi_eff)**2
        self.v_esc_equator = np.sqrt(2*GG_Q*self.mass/(self.radi_eff)).si

    def set_optical(self, hmag_vis=None, slope_par=None, diam_eff=None,
                    p_vis=None, a_bond=None, phase_int=None):
        p1 = solve_pAG(p_vis=p_vis, a_bond=a_bond, slope_par=slope_par)
        p2 = solve_pDH(p_vis=p_vis, diam_eff=diam_eff, hmag_vis=hmag_vis)
        p3 = solve_Gq(slope_par=slope_par, phase_int=phase_int)

        try:
            np.testing.assert_allclose(p1["p_vis"], p2["p_vis"])
        except AssertionError:
            raise AssertionError(
                "The p_vis values obtained using the relations of "
                + "[p_vis, a_bond, slope_par] and "
                + "[p_vis, diam_eff, hmag_vis] are different. "
                + "Please check the input values.")

        try:
            np.testing.assert_allclose(p1["slope_par"], p3["slope_par"])
        except AssertionError:
            raise AssertionError(
                "The slope_par values obtained using the relations of "
                + "[p_vis, a_bond, slope_par] and "
                + "[slope_par, phase_int] are different. "
                + "Please check the input values.")

        ps = p1.copy()
        ps.update(p2)
        ps.update(p3)

        for p in ["p_vis", "a_bond", "slope_par", "diam_eff",
                  "hmag_vis", "phase_int"]:
            if getattr(self, p) is not None:
                try:
                    u.allclose(getattr(self, p), ps[p])
                except AssertionError:
                    warn(f"self.{p} is not None ({getattr(self, p)}), "
                         + f"and will be overridden by {ps[p]}.")

        self.p_vis = ps["p_vis"]
        self.a_bond = ps["a_bond"]
        self.slope_par = ps["slope_par"]
        self.diam_eff = ps["diam_eff"]
        self.radi_eff = self.diam_eff/2
        self.hmag_vis = ps["hmag_vis"]
        self.phase_int = ps["phase_int"]

    def set_thermal(self, ti, emissivity, eta_beam=1):
        toQ = dict(return_quantity=True)
        ps1 = solve_temp_eqm(temp_eqm=None,
                             a_bond=self.a_bond,
                             eta_beam=eta_beam,
                             r_hel=self.r_hel,
                             emissivity=emissivity,
                             **toQ)
        self.emissivity = ps1["emissivity"]
        self.eta_beam = ps1["eta_beam"]
        self.temp_eqm = ps1["temp_eqm"]
        self.temp_eqm_1au = np.sqrt(self.r_hel.to(u.au).value) * self.temp_eqm
        self.temp_eqm__K = (self.temp_eqm.to(u.K)).value

        ps2 = solve_thermal_par(thermal_par=None,
                                ti=ti,
                                rot_period=self.rot_period,
                                temp_eqm=self.temp_eqm,
                                emissivity=emissivity,
                                **toQ)
        self.thermal_par = ps2["thermal_par"]
        self.ti = ps2["ti"]

    # Currently due to the spline, nlat must be > 1
    def set_tpm(self, nlon=360, nlat=90, Zmax=10, nZ=50):
        ''' TPM code related parameters.
        The attributes here are all non-dimensional!
        '''
        self.nlon = nlon
        if nlat < 3:
            warn("Currently nlat < 3 is not supported. "
                 + "Internally I will use nlat = 3.")
            self.nlat = 3
        else:
            self.nlat = nlat
        self.dlon = 2*PI/self.nlon
        self.dlat = PI/self.nlat
        self.Zmax = Zmax
        self.nZ = nZ
        self.dZ = self.Zmax/self.nZ

        if (self.dlon/(self.dZ)**2) > 0.5:
            raise ValueError("dlon/dZ^2 > 0.5 !!"
                             "The solution may not converge. Decrease depth"
                             " resolution (nZ) or increase time resolution "
                             "(nlon).")

    def minimal_set(self, thermal_par, aspect_ang, temp_eqm=1):
        ''' Set minimal model.
        Note
        ----
        When we just want to calculate the temperature on the asteroid,
        not thinking about the viewing geometry, there is no need to
        consider any detail about the spin. The spin direction is
        absorbed into the aspect angle. The spin period is absorbed into
        thermal paramter. Diameter does not affect the temperature,
        unless we are using the p-D-H relation.
        '''
        toQ = dict(to_value=False)
        self.thermal_par = change_to_quantity(thermal_par, NOUNIT, **toQ)
        self.aspect_ang = change_to_quantity(aspect_ang, u.deg, **toQ)
        self.temp_eqm = change_to_quantity(temp_eqm, u.K, **toQ)
        self.temp_eqm__K = (self.temp_eqm.to(u.K)).value

        # set fictitious ephemerides
        self.r_hel_vec = np.array([1, 0, 0])
        self.spin_vec = np.array([-np.cos(self.aspect_ang).value,
                                  0,
                                  np.sin(self.aspect_ang).value])

    def calc_temp(self, full=False, min_iter=50, permanent_shadow_u=0):
        ''' Calculate the temperature using TPM
        Parameters
        ----------
        full : bool, optional.
            If ``True``, the temperature beneath the surface is also
            saved as ``self.tempfull`` as well as the surface temperatue
            as ``self.tempsurf``. If ``False`` (default), only the
            surface tempearatue is saved as ``self.tempsurf``.
        '''
        phases = np.arange(0, 2*PI, self.dlon)*u.rad
        # colats is set s.t. nlat=1 gives colat=90 deg.
        colats = np.arange(0 + self.dlat/2, PI, self.dlat)*u.rad
        Zarr = np.arange(0, self.Zmax, self.dZ)

        # For interpolation in lon = 360 deg - dlon to 360 deg:
        phases_spl = np.arange(0, 360 + self.dlon*R2D, self.dlon*R2D)
        colats_spl = (colats.to(u.deg)).value

        # Make nlon + 1 and then remove this last element later
        u_arr = np.zeros(shape=(self.nlat, self.nlon + 1, self.nZ))

        # initial guess = temp_eqm*e^(-depth/skin_depth)
        for k in range(self.nlat):
            u_arr[k, 0, :] = np.exp(-Zarr)

        self.mu_suns = calc_mu_suns(r_hel_vec=self.r_hel_vec,
                                    spin_vec=self.spin_vec,
                                    phases=phases,
                                    colats=colats,
                                    full=False)

        # For interpolation in lon = 360 deg - dlon to 360 deg:
        _mu_suns = self.mu_suns.copy()
        _mu_suns = np.append(_mu_suns, np.atleast_2d(_mu_suns[:, 0]).T, axis=1)
        self.spl_musun = RectBivariateSpline(colats_spl, phases_spl, _mu_suns,
                                             kx=1, ky=1, s=0)

        if self.thermal_par.value < 1.e-6:
            warn("Thermal parameter too small: Automatically set to 1.e-6.")
            self.thermal_par = 1.e-6*NOUNIT

        setup_uarr_tpm(u_arr,
                       thpar=self.thermal_par.value,
                       dlon=self.dlon,
                       dZ=self.dZ,
                       mu_suns=self.mu_suns,
                       min_iter=min_iter,
                       permanent_shadow_u=permanent_shadow_u)

        # Because there is one more "phase" value, we make spline here
        # before erasing it in the next line:
        self.spl_temp = RectBivariateSpline(colats_spl,
                                            phases_spl,
                                            u_arr[:, :, 0],
                                            kx=1, ky=1, s=0)

        # because there is one more "phase" value, erase it:
        u_arr = u_arr[:, :-1, :]

        self.tempunit = "T_EQM"
        self.tpm_colats = (colats.to(u.deg)).value
        self.tpm_phases = (phases.to(u.deg)).value

        if full:
            self.tempfull = u_arr
            self.tempsurf = u_arr[:, :, 0]
        else:
            self.tempsurf = u_arr[:, :, 0]

    def get_temp(self, colat__deg, lon__deg):
        ''' Return 1d array of temperature.
        Note
        ----
        If you want 2-d array, just use ``self.temp_eqm *
        self.spl_temp(colat, phi)``.

        Parameters
        ----------
        colat__deg, lon__deg : float or Quantity, or array of such
            The colatitude, which is 0 at North and 180 at South, and
            the phase (longitude), which is 0 at midnight and 90 at
            sunrise in degrees unit. Note this is different from
            low-level cases where the default is radian in many cases.
        Note
        ----
        For performance issue, I didn't put any astropy quantity here.
        This function may be used hundreds of thousands of times for
        each simulation, so 1ms is not a small time delay.
        '''
        temp = self.spl_temp(colat__deg, lon__deg)
        return self.temp_eqm__K * temp.flatten()

    def get_musun(self, colat__deg, lon__deg):
        ''' Return 1d array of temperature.
        Note
        ----
        If you want 2-d array, just use ``self.spl_musun(colat, phi)``.

        Parameters
        ----------
        colat, phi : float or Quantity, or array of such
            The colatitude, which is 0 at North and 180 at South, and
            the phase (longitude), which is 0 at midnight and 90 at
            sunrise in degrees unit. Note this is different from
            low-level cases where the default is radian in many cases.
        Note
        ----
        For performance issue, I didn't put any astropy quantity here.
        This function may be used hundreds of thousands of times for
        each simulation, so 1ms is not a small time delay.
        '''
        musun = self.spl_musun(colat__deg, lon__deg)
        musun[musun < 1.e-4] = 0
        # 1.e-4 corresponds to incidence angle of 89.994˚
        return musun.flatten()

    def tohdul(self, output=None, dtype='float32', **kwargs):
        hdul = fits.HDUList([fits.PrimaryHDU(),
                             fits.ImageHDU(data=self.tempsurf.astype(dtype)),
                             fits.ImageHDU(data=self.mu_suns.astype(dtype))])
        hdu_0 = hdul[0]
        hdu_T = hdul[1]
        hdu_m = hdul[2]
        # names = ["T_SURF", "MU_SUN"]

        hdu_T.header["EXTNAME"] = ("T_SURF", "Extension Name")
        hdu_T.header["BUNIT"] = (self.tempunit, "Pixel unit")
        hdu_m.header["EXTNAME"] = ("MU_SUN", "Extension Name")
        hdu_m.header["BUNIT"] = ("DIMENSIONLESS", "Pixel unit")

        now = datetime.datetime.utcnow()

        # TODO: Is is better to save these to all the extensions? Or just to
        #   the Primary?
        for i, hdr in enumerate([hdu_0.header, hdu_T.header, hdu_m.header]):
            hdr["DATE-OBS"] = (str(now), "[ISO] UT time this FITS is made")

            # TPM code parameters
            hdr["RES_LON"] = (self.dlon*R2D,
                              "[deg], phase (longitude) resolution (full=360)")
            hdr["NUM_LON"] = (self.nlon,
                              "Number of phase bins (2PI/RES_LON)")
            hdr["RES_LAT"] = (self.dlat*R2D,
                              "[deg], (co-)latitude resolution")
            hdr["NUM_LAT"] = (self.nlat,
                              "Number of (co-)latitude bins")
            hdr["RES_DEP"] = (self.dZ,
                              "[thermal_skin_depth] Depth resolution")
            hdr["MAX_DEP"] = (self.Zmax,
                              "[thermal_skin_depth] Maxumum depth")
            hdr["NUM_DEP"] = (self.nZ,
                              "Number of depth bins")

            # TPM parameters
            add_hdr(hdr, "EPSILON", self.emissivity, NOUNIT,
                    "Assumed constant emissivity at thermal region")
            add_hdr(hdr, "T_EQM", self.temp_eqm, u.K,
                    "[K] Equilibrium subsolar temperature when TI=0")
            add_hdr(hdr, "T_1AU", self.temp_eqm_1au, u.K,
                    "[K] T_EQM at r_hel=1AU")
            add_hdr(hdr, "T_MAXEQM", self.tempsurf.max(), u.K,
                    "[-] Maximum surface temperature in T_EQM unit")
            add_hdr(hdr, "T_MAX", self.temp_eqm__K*self.tempsurf.max(), u.K,
                    "[K] Maximum surface temperature")
            add_hdr(hdr, "TI", self.ti, TIU,
                    "[tiu] Thermal Inertia")
            add_hdr(hdr, "THETAPAR", self.thermal_par, NOUNIT,
                    "Thermal Parameter")

            # Spin-related paramters
            add_hdr(hdr, "SPIN_LON", self.spin_ecl_lon, u.deg,
                    "[deg] Spin vector, ecliptic longitude")
            add_hdr(hdr, "SPIN_LAT", self.spin_ecl_lon, u.deg,
                    "[deg] Spin vector, ecliptic latitude")
            add_hdr(hdr, "SPIN_X", self.spin_vec[0], NOUNIT,
                    "Spin vector, ecliptic X (unit vector)")
            add_hdr(hdr, "SPIN_Y", self.spin_vec[1], NOUNIT,
                    "Spin vector, ecliptic Y (unit vector)")
            add_hdr(hdr, "SPIN_Z", self.spin_vec[2], NOUNIT,
                    "Spin vector, ecliptic Z (unit vector)")
            add_hdr(hdr, "P_ROT", self.rot_period, u.h,
                    "[h] The rotational period")
            add_hdr(hdr, "OMEGAROT", self.rot_omega, 1/u.s,
                    "[rad/s] The rotational angular frequency")
            add_hdr(hdr, "ASP_ANG", self.aspect_ang, u.deg,
                    "[deg] The aspect angle")

            # Ephemerides parameters
            add_hdr(hdr, "R_HEL", self.r_hel, u.au,
                    "[au] Heliocentric distance")
            add_hdr(hdr, "R_HEL_X", self.r_hel_vec[0], u.au,
                    "[au] Sun-target vector, ecliptic X")
            add_hdr(hdr, "R_HEL_Y", self.r_hel_vec[1], u.au,
                    "[au] Sun-target vector, ecliptic Y")
            add_hdr(hdr, "R_HEL_Z", self.r_hel_vec[2], u.au,
                    "[au] Sun-target vector, ecliptic Z")

            add_hdr(hdr, "HECL_LON", self.hel_ecl_lon, u.deg,
                    "[deg] Sun-target vector, ecliptic longitude")
            add_hdr(hdr, "HECL_LAT", self.hel_ecl_lat, u.deg,
                    "[deg] Sun-target vector, ecliptic latitude")

            add_hdr(hdr, "R_OBS", self.r_obs, u.au,
                    "[au] Geocentric(observer-target) distance")
            add_hdr(hdr, "R_OBS_X", self.r_obs_vec[0], u.au,
                    "[au] Observer-target vector, ecliptic X")
            add_hdr(hdr, "R_OBS_Y", self.r_obs_vec[1], u.au,
                    "[au] Observer-target vector, ecliptic Y")
            add_hdr(hdr, "R_OBS_Z", self.r_obs_vec[2], u.au,
                    "[au] Observer-target vector, ecliptic Z")

            add_hdr(hdr, "OECL_LON", self.obs_ecl_lon, u.deg,
                    "[deg] Observer-target vector, ecliptic longitude")
            add_hdr(hdr, "OECL_LAT", self.obs_ecl_lat, u.deg,
                    "[deg] Observer-target vector, ecliptic latitude")

            # Albedo-size-magnitude
            add_hdr(hdr, "ALB_GEOM", self.p_vis, NOUNIT,
                    "Geometric albedo in V-band")
            add_hdr(hdr, "ALB_BOND", self.a_bond, NOUNIT,
                    "Bond albedo, e.g., (0.286+0.656*SLOPEPAR)*ALB_GEOM")
            add_hdr(hdr, "SLOPEPAR", self.slope_par, NOUNIT,
                    "Slope parameter for IAU HG mag system")
            add_hdr(hdr, "ABS_MAG", self.hmag_vis, NOUNIT,
                    "[mag] Absolute magnitude in V-band")
            add_hdr(hdr, "DIAM_EFF", self.diam_eff, u.km,
                    "[km] Effective diameter (twice RADI_EFF)")
            add_hdr(hdr, "RADI_EFF", self.radi_eff, u.km,
                    "[km] Effective radius (half DIAM_EFF)")

            # WCS: image XY -> Longitude (0, 360)/Latitude (-90, +90)
            hdr["CTYPE1"] = ("LINEAR",
                             "Coordinate unit")
            hdr["CTYPE2"] = ("LINEAR",
                             "Coordinate unit")
            hdr["CNAME1"] = ("Latitude",
                             "[deg] 90 = north pole")
            hdr["CNAME2"] = ("Longitude",
                             "[deg] 180 = noon")
            hdr["CUNIT1"] = ("deg",
                             "Coordinate unit")
            hdr["CUNIT2"] = ("deg",
                             "Coordinate unit")
            hdr["CRPIX1"] = (1,
                             "Pixel coordinate of reference point")
            hdr["CRPIX2"] = (1,
                             "Pixel coordinate of reference point")
            hdr["CRVAL1"] = (self.dlon/2*R2D,
                             "Coordinate value at reference point")
            hdr["CRVAL2"] = (-90 + self.dlat/2*R2D,
                             "Coordinate value at reference point")
            hdr["CD1_1" ] = (self.dlon*R2D,
                             "Coordinate transformation matrix element")
            hdr["CD1_2" ] = (0,
                             "Coordinate transformation matrix element")
            hdr["CD2_1" ] = (0,
                             "Coordinate transformation matrix element")
            hdr["CD2_2" ] = (self.dlat*R2D,
                             "Coordinate transformation matrix element")

            # TODO: put LON-TMAX, LAT-TMAX, LON-SS, LAT-SS, LON-SO, LAT-SO
        if output is not None:
            print(type(hdul))
            print(type(hdul[0]))
            hdul.writeto(output, **kwargs)
        return hdul
