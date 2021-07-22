'''
xyz such that
z: spin axis
x: s.t. the Sun is always on the xz plane with negative x
(sun-asteroid vector has positive x value)
longitude phi = 0 at midnight, 180˚ at midday.
'''
import numpy as np
from astropy import units as u

from .constants import C_F_SUN, C_F_THER, D2R, GG, PI, T_SUN
from .relations import solve_rmrho
from .util import cart2sph, change_to_quantity, sph2cart

__all__ = ["MovingParticle"]


# For easier handling of cart2sph, because input gets degrees of theta/phi and phi should be 0 to
# 360...
CART2SPH_KW = dict(from_0=True, degree=True, to_lonlat=False)


def sin_deg(x):
    return np.sin(x*D2R)


def cos_deg(x):
    return np.cos(x*D2R)


class SmallBodyForceMixin:
    @staticmethod
    def _force_sun(r_um, r_hel_au, a_bond, height_par, mu_sun,
                   val_Qprbar):
        f_sun = C_F_SUN*(r_um/r_hel_au)**2*val_Qprbar
        f_ref = a_bond*height_par*mu_sun*f_sun
        return f_sun, f_ref

    # TODO: Will it be faster if we have spline of temp^4 a priori?
    @staticmethod
    def _force_ther(r_um, temp, height_par, emissivity, val_Qprbar):
        r'''
        Constant emissivity for surface is assumed, i.e., :math: `\epsilon_S = \bar{\epsilon_S}`.
        '''
        eda2 = emissivity*height_par*(r_um)**2
        f_ther = C_F_THER*val_Qprbar*eda2*temp**4

        return f_ther

    # TODO: Maybe gala integrator can be used?
    # https://gala-astro.readthedocs.io/en/latest/integrate/
    # @staticmethod
    # def leapfrog_dkd(t, posvel, dt, acc_func):
    #     # t is not used...
    #     pos, vel = posvel
    #     pos_tmp = pos + dt/2*vel

    #     vel_new = vel + acc_func(pos_tmp, append=True)*dt
    #     pos_new = pos_tmp + dt/2*vel_new
    #     return np.array([pos_new, vel_new])

    @staticmethod
    def leapfrog_kdk(t, posvel, dt, acc_func, acc_start=None):
        '''
        To save time in acceleration calculation, if you give ``acc_start``, it will use it as the
        acceleration of the starting point, which is the accelration of the ending point of the last
        iteration.
        '''
        pos, vel = posvel
        if acc_start is not None:
            vel_tmp = vel + dt/2*acc_start
        else:
            vel_tmp = vel + dt/2*acc_func(pos)

        pos_new = pos + dt*vel_tmp
        vel_new = vel_tmp + dt/2*acc_func(pos_new, append=True)
        return pos_new, vel_new


class MovingParticle(SmallBodyForceMixin):
    def __init__(self, smallbody, radius=None, mass=None, mass_den=None, r0_radius=0.01):
        '''
        Parameters
        ----------
        radius, mass, mass_den : float or `~astropy.Quantity`
            The radius, mass, and mass density of the particle. At least two of them must be given to
            solve the mass-radius-density equation for a spherical particle.

        r0_radius : float, optional
            The r0 parameter (the maximum radius of the regolith to be considered for reflected solar
            radiation and thermal radiation) in the unit of the smallbody's radius.
        '''
        self.smallbody = smallbody
        self.r_sb = self.smallbody.diam_eff.to(u.m)/2
        self.r_sb_m = (self.r_sb.to(u.m)).value
        self.m_sb = self.smallbody.mass
        self.m_sb_kg = (self.m_sb.to(u.kg)).value

        self.r0_par = self.r_sb*r0_radius
        self.r0_par_m = (self.r0_par.to(u.m)).value
        self.r_hel = self.smallbody.r_hel
        self.r_hel_au = (self.r_hel.to(u.au)).value
        self.f_sun_dir = np.array([np.sin(self.smallbody.aspect_ang).value,
                                   0,
                                   -np.cos(self.smallbody.aspect_ang).value
                                   ])  # unit vector of sun-asteroid vector

        def _get_temp(theta__deg, phi__deg):
            return self.smallbody.get_temp_1d(colat__deg=theta__deg, lon__deg=phi__deg)

        def _get_musun(theta__deg, phi__deg):
            return self.smallbody.get_musun(colat__deg=theta__deg, lon__deg=phi__deg)

        self.get_temp_1d = _get_temp
        self.get_musun = _get_musun

        ps = solve_rmrho(radius=radius, mass=mass, mass_den=mass_den)
        self.radius = ps["radius"]  # m
        self.radius_um = (self.radius.to(u.um)).value  # float in um
        self.mass = ps["mass"]  # kg
        self.mass_kg = (self.mass.to(u.kg)).value
        self.mass_den = ps["mass_den"]  # kg/m^3
        self.vel_eq = 2*PI*self.r_sb/self.smallbody.rot_period
        self.vel_eq_mps = (self.vel_eq.to(u.m/u.s)).value

    def set_func_Qprbar(self, func_Qprbar, func_Qprbar_sun=None):
        '''
        Parameters
        ----------
        func_Qprbar : function object
            The function which returns Qprbar value for a given pair of temperature and radius in
            Kelvins and micro-meters units, respectively. It **must** get inputs in the order of
            temperature and radius (at least currently).

        func_Qprbar_sun : function object, optional
            The function which returns Qprbar value for the solar spectrum for a given radius in
            micro-meters unit.
        '''
        self.func_Qprbar = func_Qprbar
        self.func_Qprbar_sun = func_Qprbar_sun

        if self.func_Qprbar_sun is None:
            def _func_Qprbar_sun(r_um):
                return self.func_Qprbar(T_SUN, r_um)
            print(f"Getting Qprbar_sun function from func_Qprbar with T={T_SUN}.")
            self.func_Qprbar_sun = _func_Qprbar_sun

        # Assign a single scalar value for Qprbar_sun:
        self.val_Qprbar_sun = self.func_Qprbar_sun(self.radius_um)

    def set_initial_pos(self, colat, lon, height=1*u.cm, vec_vel_init=None):
        ''' Sets the initial position

        Parameters
        ----------
        colat, lon : float or `~astropy.Quantity`
            The colatitude (``theta`` of ``(r, theta, phi)`` notation) and the longitude (``phi`` of
            the notation, which is 0 at midnight, 90 deg at sunrise) of the initial position.
        '''
        height_init = change_to_quantity(height, u.m, to_value=False)
        heignt_init_m = (height_init.to(u.m)).value
        r = self.r_sb_m + heignt_init_m
        th = change_to_quantity(colat, u.deg, to_value=True)
        ph = change_to_quantity(lon, u.deg, to_value=True)

        # Only one-time usage so use this slow conversion function:
        self.trace_pos_xyz = [sph2cart(theta=th, phi=ph, r=r, degree=True)]
        self.trace_time = [0]
        self.trace_pos_sph = [np.array([r, th, ph])]
        if vec_vel_init is None:
            vec_vel_init = (self.vel_eq_mps*sin_deg(th)*np.array([-sin_deg(ph), cos_deg(ph), 0]))
        self.trace_vel_xyz = [np.array(vec_vel_init)]
        self.trace_rvec = [np.array(self.trace_pos_xyz[0])/r]
        self.trace_height = [heignt_init_m]
        self.trace_heightpar = [1/(1 + (heignt_init_m/self.r0_par_m)**2)]
        self.trace_musun = [self.get_musun(th, ph)]
        self.trace_temp = [self.get_temp_1d(th, ph)]

        self.trace_a_sun_xyz = []
        self.trace_a_ref_xyz = []
        self.trace_a_ther_xyz = []
        self.trace_a_grav_xyz = []
        self.trace_a_all_xyz = []
        self.acc_func(self.trace_pos_xyz[0], append=True)

    def acc_func(self, pos_xyz, append=False):
        '''
        Parameters
        ----------
        pos_xyz : 1-d array of float
            The position in XYZ format in meters units. For performance issue, it's recommended to use
            float than `~astropy.Quantity`.
        '''
        r_sph, th, ph = cart2sph(*pos_xyz, **CART2SPH_KW)

        height_par = 1/(1 + ((r_sph - self.r_sb_m)/self.r0_par_m)**2)

        mu_sun = self.get_musun(th, ph)
        temp_s = self.get_temp_1d(th, ph)
        unit_r = pos_xyz/r_sph

        val_Qprbar_surf = self.func_Qprbar(temp_s, self.radius_um)
        # NOTE: self.func_Qprbar MUST get temperature and radius in this order, and with units of
        #   Kelvins and micrometers, respectively.

        _ang_psi = np.arccos(np.dot(unit_r, self.f_sun_dir))
        in_shadow = ((_ang_psi < PI/2) and ((r_sph*np.sin(_ang_psi)) < self.r_sb_m))
        if in_shadow:
            f_sun = 0
            f_ref = 0
        else:
            f_sun, f_ref = self._force_sun(r_um=self.radius_um,
                                           r_hel_au=self.r_hel_au,
                                           a_bond=self.smallbody.a_bond.value,
                                           height_par=height_par,
                                           mu_sun=mu_sun,
                                           val_Qprbar=self.val_Qprbar_sun)
        f_ther = self._force_ther(r_um=self.radius_um,
                                  temp=temp_s,
                                  height_par=height_par,
                                  emissivity=self.smallbody.emissivity.value,
                                  val_Qprbar=val_Qprbar_surf)

        a_sun_vec = (f_sun/self.mass_kg)*self.f_sun_dir
        a_ref_vec = (f_ref/self.mass_kg)*unit_r
        a_ther_vec = (f_ther/self.mass_kg)*unit_r
        a_grav_vec = -(GG*self.m_sb_kg/r_sph**2)*unit_r
        a_all_vec = a_sun_vec + a_ref_vec + a_ther_vec + a_grav_vec

        if append:
            self.trace_a_sun_xyz.append(a_sun_vec)
            self.trace_a_ref_xyz.append(a_ref_vec)
            self.trace_a_ther_xyz.append(a_ther_vec)
            self.trace_a_grav_xyz.append(a_grav_vec)
            self.trace_a_all_xyz.append(a_all_vec)
            # self.trace_rvec.append(unit_r)

        return a_all_vec

    def _propagate(self, dt):
        ''' Propagate the particle with one single step.

        Parameters
        ----------
        dt : float
            The time step in real absolute physical unit (seconds).
        '''
        newpos_xyz, newvel_xyz = self.leapfrog_kdk(
            t=None,
            posvel=[self.trace_pos_xyz[-1], self.trace_vel_xyz[-1]],
            dt=dt,
            acc_func=self.acc_func,
            acc_start=self.trace_a_all_xyz[-1]
        )
        # TODO: Maybe put cart2sph at the ``wrapup``?
        newpos_sph = cart2sph(*newpos_xyz, **CART2SPH_KW)
        height = newpos_sph[0] - self.r_sb_m
        height_par = 1/(1 + (height/self.r0_par_m)**2)
        self.trace_time.append(self.trace_time[-1] + dt)
        self.trace_pos_xyz.append(newpos_xyz)
        self.trace_rvec.append(np.array(newpos_xyz)/newpos_sph[0])
        self.trace_pos_sph.append(newpos_sph)
        self.trace_vel_xyz.append(newvel_xyz)
        self.trace_height.append(height)
        self.trace_heightpar.append(height_par)
        self.trace_musun.append(self.get_musun(newpos_sph[1], newpos_sph[2]))
        self.trace_temp.append(self.get_temp_1d(newpos_sph[1], newpos_sph[2]))
        # NOTE: some of these append are not included in the acc_func, because if we use leapfrog_dkd,
        #   the times at which we calculate the acceleration must be different form those we use for
        #   calculate the positions. For this reason, I sacrificed a bit of (computational) time and
        #   put these code lines here, not in the acc_func.

    def propagate(self, dt, nstep=None, min_height=0*u.m, max_height=None, verbose=True):
        ''' Propagate the particle
        Parameters
        ----------
        dt : float
            The time step in real absolute physical unit (seconds).

        nstep : int or None
            The number of steps to propagate. If `None`, it is halted only when the min_height or
            max_height is reached.

        min_height, max_height : float, `~astropy.Quantity`
            The minimum and maximum height value to halt the calculation. If the particle's height
            reaches these values, the calculation (propagation) halts with ``self.halt_code`` of
            ``"min_height"`` or ``"max_height"``. Interpreted as meters unit if float.
        '''
        check_min = False
        if min_height is not None:
            check_min = True
            self.min_height = change_to_quantity(min_height, u.m, to_value=False)
            self.min_height_m = (self.min_height.to(u.m)).value

        check_max = False
        if max_height is not None:
            check_max = True
            self.max_height = change_to_quantity(max_height, u.m, to_value=False)
            self.max_height_m = (self.max_height.to(u.m)).value

        self.halt_code = None
        self.halt_code_str = None
        i = 0
        while True:
            self._propagate(dt)
            i += 1
            if check_min:
                if self.trace_height[-1] < self.min_height_m:
                    if verbose:
                        print(f"Halted ({i}-th iter): height < {self.min_height}.")
                    self.halt_code_str = "min_height"
                    self.halt_code = 1
                    self.halt_iter = i
                    break

            if check_max:
                if self.trace_height[-1] > self.max_height_m:
                    if verbose:
                        print(f"Halted ({i}-th iter): height > {self.max_height}.")
                    self.halt_code_str = "max_height"
                    self.halt_code = 2
                    self.halt_iter = i
                    break

            if nstep is not None:
                if i >= nstep:
                    break

    def wrapup(self):
        ''' Changes meaningful lists to numpy array
        '''
        _alltraces = ["trace_pos_xyz", "trace_pos_sph", "trace_vel_xyz",
                      "trace_height", "trace_heightpar", "trace_musun",
                      "trace_temp", "trace_time", "trace_rvec",
                      "trace_a_sun_xyz", "trace_a_ther_xyz",
                      "trace_a_grav_xyz", "trace_a_ref_xyz", "trace_a_all_xyz"
                      ]
        for attr in _alltraces:
            setattr(self, attr, np.array(getattr(self, attr)))

            # NOTE: trace_a_sun is nothing but a constant scalar for all
            #   the time, but just for the consistency, I let it make a
            #   1-d ndarray for it too.
            self.trace_speed = np.linalg.norm(self.trace_vel_xyz, axis=1)
            self.trace_a_sun = np.linalg.norm(self.trace_a_sun_xyz, axis=1)
            self.trace_a_ref = np.linalg.norm(self.trace_a_ref_xyz, axis=1)
            self.trace_a_ther = np.linalg.norm(self.trace_a_ther_xyz, axis=1)
            self.trace_a_grav = np.linalg.norm(self.trace_a_grav_xyz, axis=1)
