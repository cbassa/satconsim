#!/usr/bin/env python3
import yaml
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import wcs
import warnings

from satconsim.constants import d2r, r2d, rearth
from satconsim.ddvmap import compute_ddv_map, initialize_azel_map
from satconsim.utils import solar_azel, solar_illumination
from satconsim.plot import plot_map

if __name__ == "__main__":
    # Set warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)
    warnings.filterwarnings("ignore", category=UserWarning, append=True)

    # Settings
    lat = -30.244639 # Observer latitude [deg]
    elev = 2663 # Observer elevation [m]
    sundec = -0.0
    sunha = 125.3
    afov = 100 # Field-of-view [deg^2]
    texp = 60 # Exposure time [sec]
    fname = "yaml/starlink_gen2_oneweb_phase2.yaml"

    lat = -24.627222 # Observer latitude [deg]
    elev = 2635 # Observer elevation [m]
    sundec = -0.0
    sunha = 90
    afov = (6 / 60)**2 # Field-of-view [deg^2]
    texp = 300 # Exposure time [sec]
    instrument = "VLT/FORS2 Imaging"
    fname = "yaml/starlink_gen2_oneweb_phase2.yaml"

    lat = -24.627222 # Observer latitude [deg]
    elev = 2635 # Observer elevation [m]
    sundec = -0.0
    sunha = 90
    afov = (6 / 60) * (2 / 3600) # Field-of-view [deg^2]
    texp = 1800 # Exposure time [sec]
    instrument = "VLT/FORS2 spectroscopy"
    fname = "yaml/starlink_gen2_oneweb_phase2.yaml"

    lat = -24.627222 # Observer latitude [deg]
    elev = 2635 # Observer elevation [m]
    sundec = -0.0
    sunha = 90
    afov = (10 / 3600) * (2 / 3600) # Field-of-view [deg^2]
    texp = 1800 # Exposure time [sec]
    instrument = "VLT/UVES spectroscopy"
    fname = "yaml/starlink_gen2_oneweb_phase2.yaml"

    lat = -24.627222 # Observer latitude [deg]
    elev = 2635 # Observer elevation [m]
    sundec = -0.0
    sunha = 90
    afov = 1 # Field-of-view [deg^2]
    texp = 300 # Exposure time [sec]
    instrument = "VST/OmegaCAM imaging"
    fname = "yaml/starlink_gen2_oneweb_phase2.yaml"

    # Settings
    lat = -30.244639 # Observer latitude [deg]
    elev = 2663 # Observer elevation [m]
    sundec = -0.0
    sunha = 100.0
    afov = 3 * 3 # Field-of-view [deg^2]
    texp = 300 # Exposure time [sec]
    instrument = "Rubin Observatory/LSST"
    fname = "yaml/starlink_gen2_oneweb_phase2.yaml"

    # Intialize map
    nx, ny = 181, 181
    w, rx, ry, az, el = initialize_azel_map(nx, ny)

    # Read YAML file
    with open(fname, "r") as fp:
        orbital_shells = yaml.full_load(fp)

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

    ax1.text(0.5 * nx, 1.05 * ny, "Starlink Generation 2 + OneWeb Phase 2", horizontalalignment="center")
    ax1.text(0, 0.95 * ny, instrument)
    ax1.text(0, 0.9 * ny, f"FOV = {afov:g} deg$^2$")
    ax1.text(0, 0.85 * ny, f"$t_\mathrm{{exp}} = {texp}$ s")
    plt.tight_layout()
    plt.savefig("plot.png")

