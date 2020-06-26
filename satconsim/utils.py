#!/usr/bin/env python3
import numpy as np
from satconsim.constants import rearth, wearth, flat, rau, rsun, r2d, d2r, mu

def observer_posvel(lat, elev):
    slat, clat, tlat = np.sin(lat * d2r), np.cos(lat * d2r), np.tan(lat * d2r)
    u = np.arctan((1 - flat) * tlat)
    gs = (1 - flat) * np.sin(u) + elev / (1000 * rearth) * slat
    gc = np.cos(u) + elev / (1000 * rearth) * clat
    pos = rearth * np.array([gc, 0.0, gs])
    vel = rearth * wearth * np.array([0.0, gc, 0.0])

    return pos, vel

def solve_distance(pobs, usat, rsat):
    # Observer radius
    robs = np.linalg.norm(pobs)

    # Coefficients of quadratic equation
    a = 1.0
    b = 2 * np.sum(usat * pobs[:, np.newaxis, np.newaxis], axis=0)
    c = robs ** 2 - rsat ** 2

    # Find roots
    p = np.sqrt(b ** 2 - 4 * a * c)
    d1 = 0.5 * (-b + p) / a
    d2 = 0.5 * (-b - p) / a

    # Select positive distance
    d = np.where(d1 >= 0, d1, d2)

    cosalpha = (rsat ** 2 - rearth ** 2 + d ** 2) / (2 * d * rsat)
    
    return d, cosalpha

def solve_nu_raan(psat, incl, rsat):
    # Extract parameters
    x, y, z = psat
    
    # Compute ra/dec
    theta = np.arctan2(y, x)

    si, ci = np.sin(incl * d2r), np.cos(incl * d2r)

    # Maximum z value
    zmax = rsat * si
    
    # North bound passes
    nun = np.arcsin(z / zmax)
    sn, cn = np.sin(nun), np.cos(nun)
    raann = theta - np.arctan2(sn * ci, cn)

    # South bound passes
    nus = np.pi - nun
    sn, cn = np.sin(nus), np.cos(nus)
    raans = theta - np.arctan2(sn * ci, cn)
    
    return nun * r2d, raann * r2d, nus * r2d, raans * r2d

def compute_velocity_map(pobs, vobs, nu, raan, incl, r):
    # Orbital velocity
    period = 2 * np.pi * np.sqrt(r * r * r / mu)
    v = 2 * np.pi * r / period
    
    # Precompute angles
    sn, cn = np.sin(nu * d2r), np.cos(nu * d2r)
    si, ci = np.sin(incl * d2r), np.cos(incl * d2r)
    sr, cr = np.sin(raan * d2r), np.cos(raan * d2r)

    # Compute position and velocity
    psat = r * np.array([cn * cr - sn * sr * ci, cn * sr + sn * cr * ci, sn * si])
    vsat = v * np.array([-sn * cr - cn * sr * ci, -sn * sr + cn * cr * ci, cn * si])
    
    # Vector differences
    dx, dy, dz = psat - pobs[:, np.newaxis, np.newaxis]
    dvx, dvy, dvz = vsat - vobs[:, np.newaxis, np.newaxis]

    # Observer properties
    r = np.sqrt(dx * dx + dy * dy + dz * dz)
    vtot = np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
    vrad = (dvx * dx + dvy * dy + dvz * dz) / r
    vang = np.sqrt(vtot * vtot - vrad * vrad)

    return vang / r * r2d

def compute_probability(b, incl):
    sb, si = np.sin(b * d2r), np.sin(incl * d2r)

    return 1 / (np.pi * np.sqrt((sb + si) * (si - sb))) * np.cos(b * d2r) * d2r

def solar_illumination(psat, sunha, sundec, sunr):
    # Compute solar position
    sh, ch = np.sin(sunha * d2r), np.cos(sunha * d2r)
    sd, cd = np.sin(sundec * d2r), np.cos(sundec * d2r)
    psun = sunr * rau * np.array([cd * ch, cd * sh, sd])

    # Position offset with satellite
    if len(psun.shape) == 1:
        dp = psun[:, np.newaxis, np.newaxis] - psat
    elif len(psun.shape) == 2:
        dp = psun[:, :, np.newaxis, np.newaxis] - psat[:, np.newaxis]

    # Distances
    r = np.sqrt(np.sum(dp ** 2, axis=0))
    rsat = np.sqrt(np.sum(psat ** 2, axis=0))

    # Angles
    asun = np.arcsin(rsun / r) * r2d
    aearth = np.arcsin(rearth / rsat) * r2d
    if len(psun.shape) == 1:
        a = np.arccos(np.sum(-dp * psat, axis=0) / (r * rsat)) * r2d
    elif len(psun.shape) == 2:
        a = np.arccos(np.sum(-dp * psat[:, np.newaxis, :, :], axis=0) / (r * rsat)) * r2d
    
    # Boolean with illuminated fraction
    illuminated = np.where(a - aearth > asun, 1, np.nan)
    
    return illuminated

def azel(ha, dec, lat):
    sh, ch = np.sin(ha * d2r), np.cos(ha * d2r)
    sd, cd, td = np.sin(dec * d2r), np.cos(dec * d2r), np.tan(dec * d2r)
    sl, cl = np.sin(lat * d2r), np.cos(lat * d2r)

    az = np.mod(np.arctan2(sh, ch * sl - td * cl) * r2d + 180, 360)
    el = np.arcsin(sl * sd + cl * cd * ch) * r2d

    return az, el

def solar_radec(mjd):
    t = (mjd - 51544.5) / 36525.0
    l0 = np.mod(280.46646 + 36000.76983 * t + 0.0003032 * t * t, 360)
    m = np.mod(357.52911 + 35999.05029 * t - 0.0001537 * t * t, 360)
    e = 0.016708634 - 0.000042037 * t - 0.0000001267 * t * t
    c = (1.914602 - 0.004817 * t - 0.000014 * t * t) * np.sin(m * d2r)
    c += (0.019993 - 0.000101 * t) * np.sin(2 * m * d2r)
    c += 0.000289 * np.sin(3 * m * d2r)

    r = 1.000001018 * (1 - e * e) / (1 + e * np.cos((m + c) * d2r))
    n = 125.04 - 1937.136 * t
    s = l0 + c + (-0.00569 - 0.00478 * np.sin(n * d2r))
    ecl = 23.43929111 + (-46.8150 * t - 0.00059 * t * t + 0.001813 * t * t * t) / 3600.0 + 0.00256 * np.cos(n* d2r)

    ra = np.arctan2(np.cos(ecl * d2r) * np.sin(s * d2r), np.cos(s * d2r))
    dec = np.arcsin(np.sin(ecl * d2r) * np.sin(s * d2r))

    return r, ra * r2d, dec * r2d

def gmst(mjd):
    t = (mjd - 51544.5) / 36525.0

    gmst = np.mod(280.46061837 + 360.98564736629 * (mjd - 51544.5) + t * t * (0.000387933 - t / 38710000), 360)

    return gmst

                  
