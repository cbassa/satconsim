#!/usr/bin/env python3
import os
import sys
import glob
import yaml
import tqdm
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.wcs import wcs

from satconsim.constants import d2r, r2d, rearth
from satconsim.ddvmap import compute_ddv_map, initialize_azel_map
from satconsim.utils import azel, solar_illumination, solar_radec, gmst
from satconsim.plot import plot_map, plot_sources

if __name__ == "__main__":
    # Set warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)
    warnings.filterwarnings("ignore", category=UserWarning, append=True)

    # Read commandline options
    parser = argparse.ArgumentParser(description="Create allsky maps with expected counts")
    parser.add_argument("-c", "--config", help="Instrument configuration [file, default: example_config.yaml]",
                        metavar="FILE", default="instruments/example_config.yaml")
    parser.add_argument("-C", "--constellation", help="Satellite constellation configuration [file, default: constellations/example_constellation.yaml]",
                        metavar="FILE", default="example_constellation.yaml")
    parser.add_argument("-t", "--tstart", help="Start time [YYYY-MM-DDTHH:MM:SS, default: 2020-06-20T21:00:00]",
                        type=str, default="2020-06-20T21:00:00")
    parser.add_argument("-d", "--delta", help="Time step size [min, default: 5.0]",
                        type=float, default=5.0)
    parser.add_argument("-n", "--number", help="Number of steps [default: 180]", 
                        type=int, default=180)
    parser.add_argument("-o", "--output", help="Output file name, use gif extention for movie [default: plot.png]", 
                        metavar="FILE", default="plot.png")
    args = parser.parse_args()

    # Check arguments
    if not os.path.exists(args.config):
        print("Observation configuration file not found.")
        sys.exit()
    if not os.path.exists(args.constellation):
        print("Satellite constellation configuration file not found.")
        sys.exit()
        
    # Read observation config
    with open(args.config, "r") as fp:
        conf = yaml.full_load(fp)
    name = conf["name"]
    site = conf["site"]
    lon = conf["longitude_deg"]
    lat = conf["latitude_deg"]
    elev = conf["elevation_m"]
    afov = conf["fov_deg_sq"]
    texp = conf["texp_s"]

    # Create results directory if absent
    if not os.path.exists("results"):
        os.makedirs("results")

    # Remove any remaining plots
    try:
        os.system("rm results/plot_????.png")
    except:
        pass

    # Intialize map
    nx, ny = 181, 181
    w, rx, ry, az, el = initialize_azel_map(nx, ny)

    # Read YAML file
    with open(args.constellation, "r") as fp:
        constellation = yaml.full_load(fp)
    # Extract orbital shells
    orbital_shells = constellation["shells"]
        
    # Number of orbital shells
    nshell = len(orbital_shells)

    # FOV radius
    rfov = np.sqrt(afov / np.pi)

    # Shell numbers
    nsatshell = np.zeros(nx * ny * nshell).reshape(nshell, nx, ny)
    nsattot = np.zeros(nx * ny).reshape(nx, ny)
    psat = np.zeros(3 * nx * ny * nshell).reshape(nshell, 3, nx, ny)
    
    # Loop over shells
    for i, o in enumerate(orbital_shells):
        print(o["name"])
        nsat = o["number_of_planes"] * o["satellites_per_plane"]
        incl = o["inclination_deg"]
        rsat = o["altitude_km"]
        
        # Compute maps
        psat[i], density, d, v = compute_ddv_map(az, el, lat, elev, incl, rsat, nsat)

        # Mask bad areas
        c = (el >= 0.0) & (np.abs(psat[i, 2]) <= (rsat + rearth) * np.sin(incl * d2r))
        d[~c] = np.nan
        v[~c] = np.nan
        density[~c] = np.nan
        
        # Compute number of satellites within the exposure for this shell
        nsatshell[i] = density * afov + 2 * density * rfov * v * texp 

        # Reset nan/infs to zero to allow summing
        c = np.isfinite(nsattot)
        nsattot[~c] = 0
        nsattot += nsatshell[i]

    # Reset points below horizon and zeroed
    c = (el >= 0) & (nsattot > 0)
    nsattot[~c] = np.nan
        
    # Compute solar position
    t = Time(args.tstart, format="isot", scale="utc") + args.delta * np.arange(args.number) * u.min
    sunr, sunra, sundec = solar_radec(t.mjd)
    lst = gmst(t.mjd) + lon
    sunha = np.mod(lst - sunra, 360)
    sunha = np.where(sunha < 180, sunha, sunha - 360)
    sunaz, sunel = azel(sunha, sundec, lat)
  
    # Loop time
    for i in tqdm.tqdm(range(len(t))):
        nsattot = np.zeros(nx * ny).reshape(nx, ny)

        # Loop over shells
        for j in range(nshell):
            # Illumination
            illuminated = solar_illumination(psat[j], sunha[i], sundec[i], 1.0)

            # Apply illumination
            nsattot += nsatshell[j] * illuminated

            # Reset nan/infs to zero to allow summing
            c = np.isfinite(nsattot)
            nsattot[~c] = 0
            
        # Reset points below horizon and zeroed
        c = (el >= 0) & (nsattot > 0)
        nsattot[~c] = np.nan

        # Extrema
        nsattotmin, nsattotmax = np.nanmin(nsattot), np.nanmax(nsattot)

        # Generate figure
        cmap = "plasma"

        fig, ax1 = plt.subplots(figsize=(8, 8))

        plot_map(fig, ax1, w, rx, ry, nsattot, nsattotmin, nsattotmax, cmap, f"Number of satellites", normtype="log")

        plot_sources(w, ax1, lat, lst[i])
        
        # Exposure settings
        ax1.text(0.5 * nx, 1.05 * ny, constellation["name"], horizontalalignment="center")
        ax1.text(0, 0.95 * ny, name)
        ax1.text(0, 0.9 * ny, f"FOV = {afov:g} deg$^2$")
        ax1.text(0, 0.85 * ny, f"$t_\mathrm{{exp}} = {texp}$ s")

        # Site info
        ax1.text(nx, 0.95 * ny, site, horizontalalignment="right")
        ax1.text(nx, 0.9 * ny, f"$\\theta={lon:.2f}^\circ$", horizontalalignment="right")
        ax1.text(nx, 0.85 * ny, f"$\phi={lat:.2f}^\circ$", horizontalalignment="right")
        ax1.text(nx, 0.8 * ny, f"$H={elev:g}$ m", horizontalalignment="right")

        # Time info
        ax1.text(0, 0, t[i].isot)

        # Solar info
        ax1.text(nx, 0.1 * ny, f"$H_\odot={sunha[i]:.2f}^\circ$", horizontalalignment="right")
        ax1.text(nx, 0.05 * ny, f"$\delta_\odot={sundec[i]:.2f}^\circ$", horizontalalignment="right")
        ax1.text(nx, 0.0 * ny, f"$A_\odot={sunel[i]:.2f}^\circ$", horizontalalignment="right")

        # Status
        if sunel[i] < -18:
            text = "Night"
        elif sunel[i] < -12:
            text = "Astronomical twilight"
        elif sunel[i] < -6:
            text = "Nautical twilight"
        elif sunel[i] < 0:
            text = "Civil twilight"
        else:
            text = "Daytime"
        ax1.text(0.5 * nx, -0.05 * ny, text, horizontalalignment="center")
        
        plt.tight_layout()
        if args.output[-3:] == "gif":
            plt.savefig(f"results/plot_{i:04d}.png")
        else:
            plt.savefig(args.output)

    # Merge plots
    if args.output[-3:] == "gif":
        cmd = f"convert results/plot_????.png {args.output}"
        os.system(cmd)
