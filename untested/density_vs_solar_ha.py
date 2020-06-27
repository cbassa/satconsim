#!/usr/bin/env python3
import yaml
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

    if np.sum(c) > 0:
        cs = ax.contour(rx, ry, v, colors=["floralwhite"], norm=norm, origin="lower")
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

    # Define arrays
    ha = np.linspace(0, 360, 100)
    dens_summer = np.zeros_like(ha)
    dens_equinox = np.zeros_like(ha)
    dens_winter = np.zeros_like(ha)
    dens_summer_tot = np.zeros_like(ha)
    dens_equinox_tot = np.zeros_like(ha)
    dens_winter_tot = np.zeros_like(ha)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Loop over shells
    for o in orbital_shells:
        print(o["name"])
        nsat = o["number_of_planes"] * o["satellites_per_plane"]
        incl = o["inclination_deg"]
        rsat = o["altitude_km"]
        
        # Compute maps
        psat, density, d, v = compute_ddv_map(az, el, lat, elev, incl, rsat, nsat)

        # Altitude cutoff
        c = el > 30
        for i, sunha in enumerate(ha):
            # Summer
            illuminated = solar_illumination(psat, sunha, -23.4, 1.0)
            dens_summer[i] = np.nansum(density[c] * illuminated[c])

            # Equinox
            illuminated = solar_illumination(psat, sunha, 0.0, 1.0)
            dens_equinox[i] = np.nansum(density[c] * illuminated[c])

            # Equinox
            illuminated = solar_illumination(psat, sunha, 23.4, 1.0)
            dens_winter[i] = np.nansum(density[c] * illuminated[c])

        # Add to total
        dens_summer_tot += dens_summer
        dens_equinox_tot += dens_equinox
        dens_winter_tot += dens_winter

        if "Starlink" in o["name"]:
            color="C0"
        if "OneWeb" in o["name"]:
            color="C1"    
        
        ax1.plot(ha / 15, dens_winter, color)
        ax2.plot(ha / 15, dens_equinox, color)
        ax3.plot(ha / 15, dens_summer, color)

        
    ax1.plot(ha / 15, dens_winter_tot, "k")
    ax2.plot(ha / 15, dens_equinox_tot, "k")
    ax3.plot(ha / 15, dens_summer_tot, "k")
    ax1.set_title(r"Local winter ($\delta_\odot=23.4^\circ$) at Rubin Observatory (latitude $%.2f^\circ$)" % lat)
    ax2.set_title(r"Equinox ($\delta_\odot=0^\circ$) at Rubin Observatory (latitude $%.2f^\circ$)" % lat)
    ax3.set_title(r"Local summer ($\delta_\odot=-23.4^\circ$) at Rubin Observatory (latitude $%.2f^\circ$)" % lat)
    ax3.set_xlabel("Local solar time (h)")
    ax2.set_ylabel("Number of satellites above elevations of $30^\circ$")

    ax1.text(0, 850, "Starlink generation 2", color="C0")
    ax1.text(0, 750, "OneWeb phase 2", color="C1")
    ax1.text(0, 650, "Total", color="k")
    
    #ax1.legend()
    
    plt.tight_layout()
    plt.savefig("plot_density_vs_solarha.png")
