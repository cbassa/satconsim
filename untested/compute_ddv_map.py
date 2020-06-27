#!/usr/bin/env python3
import os
import sys
import yaml
import tqdm
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import wcs

from satconsim.constants import d2r, r2d, rearth
from satconsim.ddvmap import compute_ddv_map, initialize_azel_map
from satconsim.utils import solar_azel, solar_illumination
from satconsim.plot import plot_map

if __name__ == "__main__":
    # Set warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)
    warnings.filterwarnings("ignore", category=UserWarning, append=True)

    # Read commandline options
    parser = argparse.ArgumentParser(description="Create allsky maps with expected counts")
    parser.add_argument("-c", "--config", help="Observation configuration [file, default: example_config.yaml]",
                        metavar="FILE", default="example_config.yaml")
    parser.add_argument("-C", "--constellation", help="Satellite constellation configuration [file, default: example_constellation.yaml]",
                        metavar="FILE", default="example_constellation.yaml")
    parser.add_argument("-H", "--hourangle", help="Solar hour angle [deg, default: 90.0]",
                        type=float, default=90.0)
    parser.add_argument("-d", "--declination", help="Solar declination [deg, default: 0.0]",
                        type=float, default=0.0)
    parser.add_argument("-o", "--output", help="Output file name [default: plot.png]", 
                        metavar="FILE", default="plot.png")
    args = parser.parse_args()

    # Check arguments
    if not os.path.exists(args.config):
        print("Observation configuration file not found.")
        sys.exit()
    if not os.path.exists(args.constellation):
        print("Satellite constellation configuration file not found.")
        sys.exit()
    sunha = args.hourangle
    sundec = args.declination

    # Read observation config
    with open(args.config, "r") as fp:
        conf = yaml.full_load(fp)
    name = conf["name"]
    lat = conf["latitude_deg"]
    elev = conf["elevation_m"]
    afov = conf["fov_deg_sq"]
    texp = conf["texp_s"]

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

    # Solar azimuth, elevation
    sunaz, sunel = solar_azel(sunha, sundec, lat)

    # FOV radius
    rfov = np.sqrt(afov / np.pi)
    
    # Loop over shells
    for i, o in enumerate(orbital_shells):
        print(o["name"])
        nsat = o["number_of_planes"] * o["satellites_per_plane"]
        incl = o["inclination_deg"]
        rsat = o["altitude_km"]
        
        # Compute maps
        psat, density, d, v = compute_ddv_map(az, el, lat, elev, incl, rsat, nsat)

        # Mask bad areas
        c = (el >= 0.0) & (np.abs(psat[2]) <= (rsat + rearth) * np.sin(incl * d2r))
        d[~c] = np.nan
        v[~c] = np.nan
        density[~c] = np.nan
        
        # Illumination
        illuminated = solar_illumination(psat, sunha, sundec, 1.0)

        # Compute number of satellites within the exposure for this shell
        nshell = density * afov + 2 * density * rfov * v * texp 

        # Apply illumination
        nshell *= illuminated

        # Reset nan/infs to zero to allow summing
        c = np.isfinite(nshell)
        nshell[~c] = 0

        # Add to total
        if i == 0:
            ntot = nshell
        else:
            ntot += nshell

    # Reset points below horizon and zeroed
    c = (el >= 0) & (ntot > 0)
    ntot[~c] = np.nan

    # Generate figure
    cmap = "plasma"

    fig, ax1 = plt.subplots(figsize=(8, 8))

    plot_map(fig, ax1, w, rx, ry, ntot, cmap, f"Number of satellites", normtype="log")
    ax1.text(nx, 0.95 * ny, f"$\phi={lat:.2f}^\circ$", horizontalalignment="right")
    ax1.text(nx, 0.9 * ny, f"$H_\odot={sunha:.2f}^\circ$", horizontalalignment="right")
    ax1.text(nx, 0.85 * ny, f"$\delta_\odot={sundec:.2f}^\circ$", horizontalalignment="right")
    ax1.text(nx, 0.8 * ny, f"$A_\odot={sunel:.2f}^\circ$", horizontalalignment="right")

    ax1.text(0.5 * nx, 1.05 * ny, constellation["name"], horizontalalignment="center")
    ax1.text(0, 0.95 * ny, name)
    ax1.text(0, 0.9 * ny, f"FOV = {afov:g} deg$^2$")
    ax1.text(0, 0.85 * ny, f"$t_\mathrm{{exp}} = {texp}$ s")
    plt.tight_layout()
    plt.savefig(args.output)

