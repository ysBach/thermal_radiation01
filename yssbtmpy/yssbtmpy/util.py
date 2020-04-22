import numba as nb
import numpy as np
from astropy import units as u
from numba import njit

from .constants import D2R, PI, R2D

__all__ = ["change_to_quantity", "add_hdr", "parse_obj",
           "lonlat2cart", "sph2cart", "cart2sph", "M_ec2fs", "M_bf2ss",
           "calc_mu_suns",
           "newton_iter_tpm", "setup_uarr_tpm"]


def change_to_quantity(x, desired='', to_value=False):
    ''' Change the non-Quantity object to astropy Quantity.
    Parameters
    ----------
    x : object changable to astropy Quantity
        The input to be changed to a Quantity. If a Quantity is given,
        ``x`` is changed to the ``desired``, i.e., ``x.to(desired)``.
    desired : str or astropy Unit
        The desired unit for ``x``.
    to_value : bool, optional.
        Whether to return as scalar value. If ``True``, just the
        value(s) of the ``desired`` unit will be returned after
        conversion.

    Return
    ------
    ux: Quantity

    Note
    ----
    If Quantity, transform to ``desired``. If ``desired = None``, return
    it as is. If not Quantity, multiply the ``desired``. ``desired =
    None``, return ``x`` with dimensionless unscaled unit.
    '''
    def _copy(xx):
        try:
            xcopy = xx.copy()
        except AttributeError:
            import copy
            xcopy = copy.deepcopy(xx)
        return xcopy

    try:
        ux = x.to(desired)
        if to_value:
            ux = ux.value
    except AttributeError:
        if not to_value:
            if isinstance(desired, str):
                desired = u.Unit(desired)
            try:
                ux = x*desired
            except TypeError:
                ux = _copy(x)
        else:
            ux = _copy(x)
    except TypeError:
        ux = _copy(x)
    except u.UnitConversionError:
        raise ValueError("If you use astropy.Quantity, you should use "
                         + f"unit convertible to `desired`. \nYou gave "
                         + f'"{x.unit}", unconvertible with "{desired}".')

    return ux


"""
def change_to_quantity(x, desired='', to_value=False):
    ''' Change the non-Quantity object to astropy Quantity.
    Parameters
    ----------
    x : object changable to astropy Quantity
        The input to be changed to a Quantity. If a Quantity is given,
        ``x`` is changed to the ``desired``, i.e., ``x.to(desired)``.
    desired : str or astropy Unit
        The desired unit for ``x``.
    to_value : bool, optional.
        Whether to return as scalar value. If ``True``, just the
        value(s) of the ``desired`` unit will be returned after
        conversion.

    Return
    ------
    ux: Quantity

    Note
    ----
    If Quantity, transform to ``desired``. If ``desired = None``, return
    it as is. If not Quantity, multiply the ``desired``. ``desired =
    None``, return ``x`` with dimensionless unscaled unit.
    '''
    if isinstance(x, u.Quantity):
        if isinstance(desired, str):
            desired = u.Unit(desired)

        try:
            ux = x.to(desired)
        except u.UnitConversionError:
            raise ValueError("If you use astropy.Quantity, you should use "
                             + f"unit convertible to `desired`. \nYou gave "
                             + f'"{x.unit}", unconvertible with "{desired}".')

        if to_value:
            ux = ux.value

    else:  # if no unit
        try:  # if possible, copy just in case
            ux = x.copy()
        except AttributeError:
            import copy
            ux = copy.deepcopy(x)

        if not to_value:  # if you wanted Quantity
            if ux is None:  # if a single None
                return None
            elif np.any(np.equal(ux, None)):  # If many None
                return np.empty_like(ux, dtype=object)
                # dtype=object will return ``None``, not just arbitrary values
            elif isinstance(desired, str):
                desired = u.Unit(desired)

            ux = ux * desired

    return ux
"""


def add_hdr(header, key, val, desired_unit='', comment=None):
    _val = change_to_quantity(val, desired=desired_unit, to_value=True)
    header[key] = (_val, comment)
    return header


def parse_obj(objfile):
    ''' Parses the .obj file.
    Parameters
    ----------
    objfile : path-like
        The path to the file.

    Return
    ------
    a dict object containing the raw str, vertices, facets, normals, and areas.
    '''
    objstr = np.loadtxt(objfile, dtype=bytes).astype(str)
    vertices = objstr[objstr[:, 0] == 'v'][:, 1:].astype(float)
    facets = objstr[objstr[:, 0] == 'f'][:, 1:].astype(int)

    # Normals include direction + area information
    facet_normals_ast = []
    facet_areas = []

    # I don't think we need to speed up this for loop too much since it takes
    # only ~ 1 s even for 20000 facet case.
    for facet in facets:
        verts = vertices[facet - 1]  # Python is 0-indexing!!!
        vec10 = verts[1] - verts[0]
        vec20 = verts[2] - verts[0]

        area = np.linalg.norm(np.cross(vec10, vec20)) / 2  # Triangular
        facet_com_ast = np.sum(verts, axis=0) / 3

        facet_normals_ast.append(facet_com_ast)
        facet_areas.append(area)

    facet_normals_ast = np.array(facet_normals_ast)
    facet_areas = np.array(facet_areas)

    return dict(objstr=objstr, vertices=vertices, facets=facets,
                normals=facet_normals_ast, areas=facet_areas)


def lonlat2cart(lon, lat, degree=True, r=1):
    ''' Converts the lon/lat coordinate to Cartesian coordinate.
    Parameters
    ----------
    lon, lat : float or ~astropy.Quantity
        The longitude and latitude. If float, the unit is understood
        from ``degree``.
        Note that the latitude here is not the usual "theta"
        (``theta = 90 - lat``).

    degree : bool, optional
        Whether the input ``lon, lat`` are degrees (Default) or radian
        (if ``degree=False``).

    r : float, optional.
        The radial distance from the origin. Defaults to ``1``, i.e.,
        the unit vector will be returned.

    Return
    ------
    a: 1-d array
        The calculated ``(x, y, z)`` array.
    '''
    if degree:
        targ_unit = u.deg
    else:
        targ_unit = u.rad

    lon = change_to_quantity(lon, targ_unit, to_value=False)
    lat = change_to_quantity(lat, targ_unit, to_value=False)
    theta = 90*u.deg - lat
    return sph2cart(theta=theta, phi=lon, r=r)


def sph2cart(theta, phi, degree=True, r=1):
    ''' Converts the spherical coordinate to Cartesian coordinate.
    Parameters
    ----------
    theta, phi : float or ~astropy.Quantity
        The theta and phi of the ``(r, theta, phi)`` notation. If float,
        the unit is understood from ``degree``.

    degree : bool, optional
        Whether the input ``theta, phi`` are degrees (Default) or radian
        (if ``degree=False``).

    r : float, or `~astropy.Quantity`, optional.
        The radial distance from the origin. Defaults to ``1``, i.e.,
        the unit vector will be returned.

    Return
    ------
    a: 1-d array
        The calculated ``(x, y, z)`` array.
    '''
    if degree:
        targ_unit = u.deg
    else:
        targ_unit = u.rad

    th = change_to_quantity(theta, targ_unit, to_value=False)
    ph = change_to_quantity(phi, targ_unit, to_value=False)

    sin_th = (np.sin(th)).value
    cos_th = (np.cos(th)).value
    sin_ph = (np.sin(ph)).value
    cos_ph = (np.cos(ph)).value

    x = r * sin_th * cos_ph
    y = r * sin_th * sin_ph
    z = r * cos_th
    a = np.array([x, y, z])
    return a


def cart2sph(x, y, z, from_0=True, degree=True, to_lonlat=False):
    ''' Converts the Cartesian coordinate to lon/lat coordinate
    Parameters
    ----------
    x, y, z : float
        The Cartesian (x, y, z) coordinate.

    degree : bool, optional.
        If ``False``, the returned theta and phi will be in radian. If
        ``True``(default), those will be in degrees unit.

    from_0: bool, optional
        If ``True`` (Default), the ``phi`` (or ``lon``) will be in ``0``
        to ``PI`` radian range. If ``False``, i.e., if ``phi`` (or
        ``lon``) starts from ``-PI``, it will be in ``-PI`` to ``+PI``
        range.

    Return
    ------
    a: 1-d array
        The ``(r, theta, phi)`` or ``(r, lon=phi, lat=90deg - theta)``
        array.
    '''
    r = np.sqrt(x**2 + y**2 + z**2)
    if degree:
        factor = R2D
    else:
        factor = 1.

    theta = factor*np.arccos(z/r)
    phi = factor*np.arctan2(y, x)  # -180 to +180 deg
    if from_0:
        phi = phi%(factor*2*PI)

    if to_lonlat:
        lat = factor*PI/2 - theta
        a = np.array([r, phi, lat])
    else:
        a = np.array([r, theta, phi])

    return a


def M_ec2fs(r_hel_vec, spin_vec):
    ''' The conversion matrix to convert ecliptic to frame system.
    Note
    ----
    Adopted from Sect. 2.4. of Davidsson and Rickman (2014), Icarus,
    243, 58. If ``a`` is a vector in ecliptic coordinate (in Cartesian
    (x, y, z)), ``m @ a`` will give the components of vector ``a`` in
    frame system, where ``m`` is the result of this function.

    Parameters
    ----------
    r_hel_vec : 1-d array
        The Cartesian coordinate of the asteroid in (heliocentric)
        ecliptic coordinate.

    spin_vec : 1-d array
        The Cartesian coordinate of the spin vector of the asteroid in
        ecliptic coordinate.

    Return
    ------
    m : 3-by-3 matrix
        The matrix that converts ecliptic coordinate to frame system.
    '''
    Z_fs_ec = spin_vec.copy()
    Y_fs_ec = np.cross(spin_vec, -r_hel_vec)
    X_fs_ec = np.cross(Y_fs_ec, Z_fs_ec)

    # The input rh or spin vector mignt not be unit vectors, so divide by
    # lengths to make a suitable matrix:
    X_fs_ec = X_fs_ec / np.linalg.norm(X_fs_ec)
    Y_fs_ec = Y_fs_ec / np.linalg.norm(Y_fs_ec)
    Z_fs_ec = Z_fs_ec / np.linalg.norm(Z_fs_ec)

    m1 = np.vstack([X_fs_ec, Y_fs_ec, Z_fs_ec]).T
    m = np.linalg.inv(m1)
    return m


def M_fs2bf(phase):
    ''' The conversion matrix to convert frame system to body-fixed frame.
    Note
    ----
    Adopted from Sect. 2.4. of Davidsson and Rickman (2014), Icarus,
    243, 58. If ``a`` is a vector in frame system (in Cartesian (x, y,
    z)), ``m @ a`` will give the components of vector ``a`` in
    body-fixed frame, where ``m`` is the result of this function.

    Parameters
    ----------
    phase : float
        The rotational phase (2 PI t / P_rot), in the unit of radian.

    Return
    ------
    m : 3-by-3 matrix
        The matrix that converts frame system coordinate to body-fixed
        frame.
    '''
    c = np.cos(phase)
    s = np.sin(phase)
    m = np.array([[-c, -s, 0], [s, -c, 0], [0, 0, 1]])
    return m


def M_bf2ss(colat):
    ''' The conversion matrix to convert body-fixed frame to surface system.
    Note
    ----
    Adopted from Sect. 2.4. of Davidsson and Rickman (2014), Icarus,
    243, 58. If ``a`` is vector in body-fixed frame (in Cartesian (x, y,
    z)), ``m @ a`` will give the components of vector ``a`` in surface
    system, where ``m`` is the result of this function.

    Parameters
    ----------
    colat : float or ~astropy.Quantity
        The co-latitude of the surface (in degrees unit if float).
        Co-latitude is the angle between the pole (spin) vector and the
        normal vector of the surface of interest.


    Return
    ------
    m : 3-by-3 matrix
        The matrix that converts body-fixed coordinate to surface
        system.
    '''
    colat__deg = change_to_quantity(colat, 'deg', to_value=True)

    c = np.cos(colat__deg * D2R)
    s = np.sin(colat__deg * D2R)
    m = np.array([[0, 1, 0], [-c, 0, s], [s, 0, c]])
    return m


def calc_mu_suns(r_hel_vec, spin_vec, phases, colats, full=False):
    ''' The conversion matrix to convert body-fixed frame to surface system.
    Note
    ----
    Adopted from Sect. 2.4. of Davidsson and Rickman (2014), Icarus,
    243, 58. If ``a`` is vector in body-fixed frame (in Cartesian (x, y,
    z)), ``m @ a`` will give the components of vector ``a`` in surface
    system, where ``m`` is the result of this function.

    Parameters
    ----------
    r_hel_vec, spin_vec : 1-D array
        The rh vector and spin vectors, respectively.
    phases : float or array of float or ~astropy.Quantity
        The phase values (in radian unit if floats)
    colats : float or array of float or ~astropy.Quantity
        The co-latitude of the surface (in degrees unit if float).
        Co-latitude is the angle between the pole (spin) vector and the
        normal vector of the surface of interest.

    Return
    ------
    mu_suns : 2-D array
        The mu_sun values.
    solar_dirs : 3-D array
        The direction to the Sun (``(x, y, z)`` along ``axis=2``).
        Returned only if ``full=True``.
    M1 : ndarray
        The conversion matrix to convert ecliptic to frame system, i.e.,
        the result of ``M_ec2fs(r_hel_vec=r_hel_vec,
        spin_vec=spin_vec)``. Returned only if ``full=True``.
    M2arr : ndarray
        The conversion matrix to convert frame system to body-fixed
        frame, i.e., the result of ``M_fs2bf(phase=phase)`` for all
        ``phase in phases``.
        Returned only if ``full=True``.

    M3arr : ndarray
        The conversion matrix to convert body-fixed frame to surface
        system, i.e., the result of ``M_bf2ss(colat__deg=colat)`` for
        all ``colat in colats``..
        Returned only if ``full=True``.
    '''
    colats__deg = change_to_quantity(colats, 'deg', to_value=True)
    phases__rad = change_to_quantity(phases, 'rad', to_value=True)

    M2arr = []
    M3arr = []
    solar_dirs = []
    mu_suns = []
    M1 = M_ec2fs(r_hel_vec=r_hel_vec, spin_vec=spin_vec)

    for phase in phases__rad:
        M2arr.append(M_fs2bf(phase=phase))

    for colat in colats__deg:
        M3arr.append(M_bf2ss(colat=colat))

    M2arr = np.array(M2arr)
    M3arr = np.array(M3arr)
    r_hel_unit = (r_hel_vec)/np.linalg.norm(r_hel_vec)
    for M3 in M3arr:
        solar_dirs.append(M3 @ M2arr @ M1 @ -r_hel_unit)
    solar_dirs = np.array(solar_dirs)
    mu_suns = solar_dirs.copy()[:, :, 2]  # Z component = cos i_sun
    mu_suns[mu_suns < 0] = 0

    if full:
        return mu_suns, solar_dirs, M1, M2arr, M3arr
    return mu_suns


@njit()
def newton_iter_tpm(newu0_init, newu1, thpar, dZ, mu_sun, Nmax=5000,
                    atol=1.e-8):
    ''' Root finding using Newton's method
    Parameters
    ----------
    newu0_init : float
        The first trial to the ``newu[0]`` value, i.e., the ansatz of
        ``newu[0]`` value.
    newu1 : float
        The ``newu[1]`` value (that will have been calculated before
        this function will be called).
    thpar : float
        The thermal parameter
    dZ : float
        The depth slab resolution in the thermal skin depth unit.
    mu_sun : float
        The cosine of the incident angle (zenith angle of the Sun).
    Nmax : int, optional
        The maximum number of iteration to halt the root finding.
    atol : float, optional
        If the absolute difference is smaller than ``atol``, the
        iteration will stop.
    '''
    x0 = newu0_init

    for i in range(Nmax):
        f0 = x0**4 - mu_sun - thpar / dZ * (newu1 - x0)
        slope = 4 * x0**3 + thpar / dZ
        x1 = x0 - f0 / slope

        # It is good if the iteration ends here:
        if abs(x1 - x0) < atol:
            return x1

        # Reset for next iteration
        x0 = x1
    return x1


# Tested on 15"MBP2018: speed is by ~10 times faster if parallel is used.
@njit(parallel=True)
def setup_uarr_tpm(u_arr, thpar, dlon, dZ, mu_suns, min_iter=50,
                   min_elevation_deg=0., permanent_shadow_u=0):
    '''
    Parameters
    ----------
    u_arr : 3d-array
        The u (u is used as the normalized temperature, T/T_EQM) array
        that must have been defined a priori. It must be satisfied that
        ``u_arr[i, j, k]`` is u of ``i``th colatitude, ``j``th time, and
        ``k``th depth.
        In yssbtmpy, the axis 1 (time axis) has length of ``ntime + 1``,
        so the code below will understand ``ntime = u_arr.shape[1] -
        1``.

    thpar : float
        The thermal parameter (frequently denoted by Theta).

    dlon, dZ : float
        The longitude and depth bin size in units of radian and thermal
        skin depth. ``dlon`` is identical to ``dT`` in M. Mueller (2007
        thesis)'s notation, for instance.

    min_iter : int, optional
        The minimum number of iteration for the equilibrium temperature
        calculation.

    min_elevation_deg : int or float, optional
        The minimum elevation to check whether the latitudinal slab is
        assumed as a permanently shadowed region.
        The latitudinal band is assumed to be in a permanent shadow if
        the sun is always below this elevation, and all the temperature
        on this latitude is just set as a constant given by
        ``permanent_shadow_u`` in the unit of ``temp_eqm``.

    permanent_shadow_u : float
        The temperature to be substituted for permanently shadowed
        regions (unit of ``temp_epm``).
    '''
    ncolat, ntimep1, ndepth = u_arr.shape
    ntime = ntimep1 - 1

    # For each colatitude, parallel calculation is possible!!!
    # So use numba's prange rather than range:
    for i_lat in nb.prange(ncolat):
        # Check whether the latitude is under permanent shadow.
        permanent_shadow = True
        for k in range(ntime):
            # If the sun reaches above ``min_elevation_deg``, i.e.,
            # mu_sun = cos(90 - EL_sun) > cos(90 - min_elevation_deg) at
            # least once, it's not a permanent shadow:
            if mu_suns[i_lat, k] > np.cos((90 - min_elevation_deg)*D2R):
                # If sun rises > min_elevation_deg
                permanent_shadow = False
                break

        if permanent_shadow:
            for k in nb.prange(ntime):
                for l in nb.prange(ndepth):
                    u_arr[i_lat, k, l] = permanent_shadow_u

        else:
            discrep = 1.
            for i_iter in range(5000):
                for i_t in range(ntime):
                    for i_z in range(1, ndepth - 1):
                        u_arr[i_lat, i_t + 1, i_z] = (
                            u_arr[i_lat, i_t, i_z]
                            + dlon/dZ**2
                            * (u_arr[i_lat, i_t, i_z - 1]
                               + u_arr[i_lat, i_t, i_z + 1]
                               - 2*u_arr[i_lat, i_t, i_z]
                               )
                        )
                    u_arr[i_lat, i_t + 1, -1] = u_arr[i_lat, i_t + 1, -2]
                    u_arr[i_lat, i_t + 1, 0] = newton_iter_tpm(
                        newu0_init=u_arr[i_lat, i_t, 0],
                        newu1=u_arr[i_lat, i_t + 1, 1],
                        thpar=thpar,
                        dZ=dZ,
                        mu_sun=mu_suns[i_lat, i_t]
                    )
                discrep = np.abs(u_arr[i_lat, 0, 0] - u_arr[i_lat, -1, 0])

                for i in range(ndepth):
                    u_arr[i_lat, 0, i] = u_arr[i_lat, -1, i]

                if i_iter > min_iter and discrep < 1.e-8:
                    break
