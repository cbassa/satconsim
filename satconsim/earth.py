#!/usr/bin/env python3
import numpy as np
from satconsim.constants import rearth, wearth, flat, d2r

def observer_posvel(lat, elev):
    sl, cl, tl = np.sin(lat * d2r), np.cos(lat * d2r), np.tan(lat * d2r)
    u = np.arctan((1 - flat) * tl)
    gs = (1 - flat) * np.sin(u) + elev / (1000 * rearth) * sl
    gc = np.cos(u) + elev / (1000 * rearth) * cl
    pos = rearth * np.array([gc, 0.0, gs])
    vel = rearth * wearth * np.array([0.0, gc, 0.0])

    return pos, vel
