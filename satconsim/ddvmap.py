#!/usr/bin/env python3
import numpy as np
from astropy.wcs import wcs

from satconsim.constants import rearth, r2d, d2r
from satconsim.utils import solve_distance, solve_nu_raan, compute_velocity_map
from satconsim.utils import compute_probability, observer_posvel

def initialize_azel_map(nx, ny):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = np.array([nx // 2, ny // 2])
    w.wcs.cdelt = np.array([1.0, 1.0])
    w.wcs.crval = np.array([0.0, 90.0])
    w.wcs.ctype = ["ALON-ZEA", "ALAT-ZEA"]

    # Set up binning (1x1 deg bins)
    rxmin, rxmax = 0, 180
    rymin, rymax = 0, 180
    rxbins = np.linspace(rxmin, rxmax, nx + 1)
    rybins = np.linspace(rymin, rymax, ny + 1)

    rx, ry = np.meshgrid(np.arange(nx), np.arange(ny))

    az, el = w.wcs_pix2world(np.stack((rx.ravel(), ry.ravel()), axis=-1), 1).T
    az = az.reshape(nx, -1)
    el = el.reshape(ny, -1)

    return w, rx, ry, az, el

def compute_ddv_map(az, el, lat, elev, incl, rsat, nsat):
    # Convert azimuth/elevation to hourangle/ declination
    sa, ca = np.sin((az + 180) * d2r), np.cos((az + 180) * d2r)
    sl, cl, tl = np.sin(lat * d2r), np.cos(lat * d2r), np.tan(lat * d2r)
    se, ce, te = np.sin(el * d2r), np.cos(el * d2r), np.tan(el * d2r)
    ha = np.arctan2(sa, ca * sl + te * cl) 
    dec = np.arcsin(sl * se - cl * ce * ca)
    
    sh, ch = np.sin(ha), np.cos(ha)
    sd, cd = np.sin(dec), np.cos(dec)
    usat = np.array([cd * ch, cd * sh, sd])

    # Compute observer position and velocity
    pobs, vobs = observer_posvel(lat, elev)

    # Compute distances
    d, cosalpha = solve_distance(pobs, usat, rsat + rearth)

    # Compute satellite positions
    psat = d * usat + pobs[:, np.newaxis, np.newaxis]

    # Latitude, longitude
    l = np.arctan2(psat[1], psat[0]) * r2d
    b = np.arcsin(psat[2] / (rsat + rearth)) * r2d

    # Sky area (for 1 sq deg)
    asky = (d * d2r) ** 2 / cosalpha
    ashell = 2 * np.pi * (rsat + rearth) ** 2 * np.cos(b * d2r) * d2r

    # Probabilty and density
    p = compute_probability(b, incl)
    density = asky / ashell * nsat  * p
    
    # Compute North/Southbound nu and raan values
    nun, raann, nus, raans = solve_nu_raan(psat, incl, rsat + rearth)

    # Compute North/Southbound velocity maps
    vn = compute_velocity_map(pobs, vobs, nun, raann, incl, rsat + rearth)
    vs = compute_velocity_map(pobs, vobs, nus, raans, incl, rsat + rearth)

    # Average velocity
    v = 0.5 * (vn + vs)

    return psat, density, d, v
