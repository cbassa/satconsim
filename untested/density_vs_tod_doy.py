#!/usr/bin/env python3
import yaml
import tqdm
import warnings
import os

import numpy as np
from scipy import optimize, interpolate

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import astropy.units as u
from astropy.time import Time
from astropy.io import fits

from satconsim.constants import d2r, r2d, rearth
from satconsim.ddvmap import compute_ddv_map, initialize_azel_map
from satconsim.utils import solar_azel, solar_illumination, solar_radec, gmst


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
    minalt = 30 # Minimum elevation [deg]
    
    # Intialize map
    nx, ny = 181, 181
    w, rx, ry, az, el = initialize_azel_map(nx, ny)
    
    # Read YAML file
    with open("starlink_gen2_oneweb_phase2.yaml", "r") as fp:
        orbital_shells = yaml.full_load(fp)

    # Number of orbital shells
    nshell = len(orbital_shells)

    # Time start
    t0 = Time("2020-01-01T00:00:00", format="isot", scale="utc") -  12 * u.h
    days = np.arange(0, 366, 2) * u.d
    minutes = np.arange(0, 1440, 10) * u.min
    nd, nt = len(days), len(minutes)

    tm, td = np.meshgrid(minutes, days)
    t = t0 + (td + tm).ravel()

    # Compute solar position
    sunr, sunra, sundec = solar_radec(t.mjd)
    sunha = gmst(t.mjd) - sunra
    sunaz, sunel = solar_azel(sunha, sundec, lat)

    # Use FITS file if present
    if os.path.exists("count.fits"):
        hdu = fits.open("count.fits")
        count = hdu[0].data.reshape(nd, nt)
        hdu.close()
    else:
        # Recompute map
        count = np.zeros(nt * nd)
    
        # Loop over shells
        c = el >= minalt
        for o in orbital_shells:
            print(o["name"])
            nsat = o["number_of_planes"] * o["satellites_per_plane"]
            incl = o["inclination_deg"]
            rsat = o["altitude_km"]
        
            # Compute maps
            psat, density, d, v = compute_ddv_map(az, el, lat, elev, incl, rsat, nsat)

            # Mask density below elevation limit
            density[~c] = np.nan

            for i in tqdm.tqdm(range(len(t))):
                illuminated = solar_illumination(psat, sunha[i], sundec[i], sunr[i])
                count[i] += np.nansum(density[c] * illuminated[c])

        count = count.reshape(nd, nt)
        fits.PrimaryHDU(count).writeto("count.fits", overwrite=True)
        
        
    fig, ax = plt.subplots(figsize=(8, 10))

    # Date extrema
    dmin, dmax = np.min(t0 + td), np.max(t0 + td)
    dmin = mdates.date2num(dmin.datetime)
    dmax = mdates.date2num(dmax.datetime)

    # Time extrema
    tmin, tmax = 12, 36
    txmin, txmax = 16, 8 + 24

    cmap = "inferno"
    #cmap = "viridis"
    #cmap = "plasma"
    #cmap = "magma"

    vmin, vmax = np.min(count), np.max(count)
    
    img = ax.imshow(count,
                    aspect="auto",
                    interpolation="None",
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    extent=[tmin, tmax, dmin, dmax])

    ax.yaxis_date()
    date_format = mdates.DateFormatter("%F")
    ax.yaxis.set_major_formatter(date_format)
    fig.autofmt_xdate(rotation=0, ha="center")

    ax.set_xticks(np.arange(tmin, tmax), minor=True)
    ax.set_xticks(np.arange(tmin, tmax, 3))
    labels = [r"%d$^\mathrm{h}$" % (x%24) for x in np.arange(tmin, tmax, 3)]
    ax.set_xticklabels(labels)
    ax.set_xlabel("Local time")
    ax.set_xlim(txmin, txmax)

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("bottom", size="5%", pad="8%")
    cbar = fig.colorbar(img, cax=cax, orientation="horizontal")
    cbar.set_label(f"Number of sunlit satellites above ${minalt}^\circ$ elevation")

    # Sunrise/sunset
    tx = np.linspace(tmin, tmax, nt, endpoint=False)
    ty = mdates.date2num((t0 + days).datetime)
    sunel = sunel.reshape(nd, nt)

    # Find sunset/sunrise/twilight
    for el in [0, -6, -12, -18]:
        txset = np.zeros(nd)
        txrise = np.zeros(nd)
        for i in range(nd):
            fint = interpolate.interp1d(tx, sunel[i] - el)
            txset[i] = optimize.bisect(fint, np.min(tx), 24)
            txrise[i] = optimize.bisect(fint, 24, np.max(tx))

        if el == 0:
            ax.fill_betweenx(ty, txmin, txset, color="w")
            ax.fill_betweenx(ty, txrise, txmax, color="w")
            color = "k"
        else:
            color = "gray"

        # Plot lines
        ax.plot_date(txset, ty, linestyle="-", color=color, marker=None)
        ax.plot_date(txrise, ty, linestyle="-", color=color, marker=None)

        i = int(0.89 * nd)

        if el == 0:
            ax.text(txset[i]-0.25, ty[i], "Sunset", rotation=75, color=color,
                    horizontalalignment="center", verticalalignment="center")
            ax.text(txrise[i]+0.2, ty[i], "Sunrise", rotation=-80, color=color,
                    horizontalalignment="center", verticalalignment="center")
        elif el == -6:
            ax.text(txset[i]-0.25, ty[i], "Civil", rotation=75, color=color,
                    horizontalalignment="center", verticalalignment="center")
            ax.text(txrise[i]+0.2, ty[i], "Civil", rotation=-80, color=color,
                    horizontalalignment="center", verticalalignment="center")
        elif el == -12:
            ax.text(txset[i]-0.25, ty[i], "Nautical", rotation=75, color=color,
                    horizontalalignment="center", verticalalignment="center")
            ax.text(txrise[i]+0.2, ty[i], "Nautical", rotation=-80, color=color,
                    horizontalalignment="center", verticalalignment="center")
        elif el == -18:
            ax.text(txset[i]-0.25, ty[i], "Astronomical", rotation=71, color=color,
                    horizontalalignment="center", verticalalignment="center")
            ax.text(txrise[i]+0.25, ty[i], "Astronomical", rotation=-80, color=color,
                    horizontalalignment="center", verticalalignment="center")

       
    ax.grid(alpha=0.2)
    ax.set_title("Sunlit satellites visible at Rubin Observatory\nStarlink Generation 2 (30000 satellites) + OneWeb Phase 2 (47844 satellites)")
    
    plt.tight_layout()
    plt.savefig("visibility.png", bbox_inches="tight")

        


            
