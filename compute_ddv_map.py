#!/usr/bin/env python3
import yaml
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.wcs import wcs
import warnings


from satconsim.constants import d2r, r2d, rearth
from satconsim.ddvmap import compute_ddv_map, initialize_azel_map
from satconsim.utils import solar_azel, solar_illumination


def plot_map(ax, w, rx, ry, v, cmap, label, normtype="linear"):
    c = np.isfinite(v)
    if np.sum(c) > 0:
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if normtype == "log":
            c = v > 0
            vmin = np.nanmin(v[c])
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            
            
        img = ax.imshow(v, origin="lower", interpolation=None, aspect=1,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        norm=norm)
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("bottom", size="5%", pad="8%")
        cbar = fig.colorbar(img, cax=cax, orientation="horizontal", label=label)
        
    ax.axis("off")
    plot_grid(w, ax)

    vlevels = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    if np.sum(c) > 0:
        cs = ax.contour(rx, ry, v, colors=["floralwhite"], levels=vlevels, norm=norm, origin="lower")
        ax.clabel(cs, inline=1, fontsize=10, fmt="%g")
    
    return

def plot_grid(w, ax):
    alphamin = 0.9
    # Elevations
    az = np.arange(361)
    for el in [0, 30, 60]:
        elev = el * np.ones_like(az)

        # Convert
        rx, ry = w.wcs_world2pix(np.stack((az, elev), axis=-1), 1).T

        if el > 0:
            color = "gray"
            linestyle="--"
            alpha = alphamin
        else:
            color = "k"
            linestyle="-"
            alpha = 1.0
        ax.plot(rx, ry, color, linestyle=linestyle, alpha=alpha)
    rx, ry = w.wcs_world2pix(0.0, 90.0, 1)
    ax.plot(rx, ry, "+", color="gray", alpha=alphamin)
        
    # Azimuths
    elev = np.arange(60)
    for az in [0, 45, 90, 135, 180, 225, 270, 315]:
        azimuth = az * np.ones_like(elev)

        # Convert
        rx, ry = w.wcs_world2pix(np.stack((azimuth, elev), axis=-1), 1).T

        linestyle="--"
        ax.plot(rx, ry, "gray", alpha=alphamin, linestyle=linestyle)

    # Labels
    rx, ry = w.wcs_world2pix(np.stack(([0, 90, 180, 270], -10 * np.ones(4)), axis=-1), 1).T
    for rxt, ryt, label in zip(rx, ry, ["N", "E", "S", "W"]):
        ax.text(rxt, ryt, label, horizontalalignment="center", verticalalignment="center")

if __name__ == "__main__":
    # Set warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)
    warnings.filterwarnings("ignore", category=UserWarning, append=True)

    # Settings
    lat = -30.244639 # Observer latitude [deg]
    elev = 2663 # Observer elevation [m]
    
    # Intialize map
    nx, ny = 181, 181
    w, rx, ry, az, el = initialize_azel_map(nx, ny)

    # Read YAML file
    with open("starlink_gen2_oneweb_phase2.yaml", "r") as fp:
        orbital_shells = yaml.full_load(fp)

    # Number of orbital shells
    nshell = len(orbital_shells)

    # Solar settings
    sundec = -0.0
    sunha = 125.3

    # Exposure settings
    afov = 100 # Field-of-view [deg^2]
    texp = 60 # Exposure time [sec]
    rfov = np.sqrt(afov / np.pi)

    # Solar azimuth, elevation
    sunaz, sunel = solar_azel(sunha, sundec, lat)

    # Loop over shells
    for i, o in enumerate(orbital_shells):
        print(o["name"])
        nsat = o["number_of_planes"] * o["satellites_per_plane"]
        incl = o["inclination_deg"]
        rsat = o["altitude_km"]

        # Compute maps
        psat, density, d, v = compute_ddv_map(az, el, lat, elev, incl, rsat, nsat)
        
        # Illumination
        illuminated = solar_illumination(psat, sunha, sundec, 1.0)
    
        newcount1 = density * afov
        newcount2 = 2 * density * rfov * v * texp 
        newcount1 *= illuminated
        newcount2 *= illuminated
        c = np.isfinite(newcount1)
        newcount1[~c] = 0
        c = np.isfinite(newcount2)
        newcount2[~c] = 0
        if i == 0:
            count1 = newcount1
            count2 = newcount2
        else:
            count1 += newcount1
            count2 += newcount2

    c = (el >= 0) & (count1 > 0)
    count1[~c] = np.nan
    count2[~c] = np.nan
            
    cmap = "plasma"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

    plot_map(ax1, w, rx, ry, count1 , cmap, f"Satellite density [deg$^{{-2}}$]", normtype="log")
    plot_map(ax2, w, rx, ry, count2 , cmap, f"Trail density [deg$^{{-2}}$]", normtype="log")
    ax1.text(0, 20, f"$\phi={lat:.2f}^\circ$")
    ax1.text(0, 10, f"$H_\odot={sunha:.2f}^\circ$")
    ax1.text(0, 0, f"$\delta_\odot={sundec:.2f}^\circ$")

    ax1.text(0, 190, "Starlink Generation 2 + OneWeb Phase 2")
    ax1.text(0, 170, f"FOV = {afov} deg$^2$")
    ax1.text(0, 160, f"$t_\mathrm{{exp}} = {texp}$ s")
    plt.tight_layout()
    plt.savefig("plot.png")

