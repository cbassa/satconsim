#!/usr/bin/env python3
import yaml
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.wcs import wcs
import warnings


from satconsim.constants import d2r, r2d, rearth
from satconsim.ddvmap import compute_ddv_map, initialize_azel_map
from satconsim.utils import solar_azel, solar_illumination

if __name__ == "__main__":
    # Set warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)
    warnings.filterwarnings("ignore", category=UserWarning, append=True)

    # Intialize map
    nx, ny = 181, 181
    w, rx, ry, az, el = initialize_azel_map(nx, ny)
    
    # Read YAML file
    with open("starlink_gen2_oneweb_phase2.yaml", "r") as fp:
        orbital_shells = yaml.full_load(fp)
    nshell = len(orbital_shells)
        
    lat = np.linspace(-90, 90, 50)

    colors = plt.cm.viridis(np.linspace(0, 1, len(orbital_shells)))
    colors = plt.cm.tab20c(np.linspace(0, 1, len(orbital_shells)))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    
    # Loop over shells
    for j, o in enumerate(orbital_shells):
        print(o["name"])
        nsat = o["number_of_planes"] * o["satellites_per_plane"]
        incl = o["inclination_deg"]
        rsat = o["altitude_km"]

        dens_low = np.zeros_like(lat)
        dens_mid = np.zeros_like(lat)
        dens_high = np.zeros_like(lat)
        dens_tot = np.zeros_like(lat)
        for i in tqdm.tqdm(range(len(lat))):
            # Compute maps
            psat, density, d, v = compute_ddv_map(az, el, lat[i], 0, incl, rsat, 1)

            # Altitude cutoff
            c = el >= 0

            # Total
            dens_tot[i] = np.nansum(density[c])
            
            # Low
            illuminated = solar_illumination(psat, 180, -23.4, 1.0)
            dens_low[i] = np.nansum(density[c] * illuminated[c])

            # Mid
            illuminated = solar_illumination(psat, 180, 0, 1.0)
            dens_mid[i] = np.nansum(density[c] * illuminated[c])

            # High
            illuminated = solar_illumination(psat, 180, 23.4, 1.0)
            dens_high[i] = np.nansum(density[c] * illuminated[c])

        ax1.plot(lat, dens_tot, color=colors[j])
        ax2.plot(lat, dens_mid, color=colors[j])
        ax3.plot(lat, dens_low, color=colors[j])
        ax4.plot(lat, dens_high, color=colors[j])
        
    ax1.set_ylim(0, 0.2)
    ax2.set_ylim(0, 0.2)
    ax3.set_ylim(0, 0.2)
    ax4.set_ylim(0, 0.2)
    ax1.set_xlim(-90, 90)
    ax2.set_xlim(-90, 90)
    ax3.set_xlim(-90, 90)
    ax4.set_xlim(-90, 90)
    ax1.set_title("Sunset/Sunrise")
    ax2.set_title("Midnight at Equinox")
    ax3.set_title(r"Midnight at December Solstice")
    ax4.set_title(r"Midnight at June Solstice")

    ax3.set_xlabel(r"Latitude ($^\circ$)")
    ax4.set_xlabel(r"Latitude ($^\circ$)")
    ax1.set_ylabel("Visible fraction of constellation")
    ax3.set_ylabel("Visible fraction of constellation")

    ax1.set_xticks(range(-90, 91, 30))
    ax1.set_xticks(range(-90, 91, 10), minor=True)
    ax1.set_yticks(np.arange(0.0, 0.21, 0.05))
    ax1.set_yticks(np.arange(0.0, 0.21, 0.01), minor=True)
    ax1.grid(alpha=0.2)
    ax2.set_xticks(range(-90, 91, 30))
    ax2.set_xticks(range(-90, 91, 10), minor=True)
    ax2.set_yticks(np.arange(0.0, 0.21, 0.05))
    ax2.set_yticks(np.arange(0.0, 0.21, 0.01), minor=True)
    ax2.grid(alpha=0.2)
    ax3.set_xticks(range(-90, 91, 30))
    ax3.set_xticks(range(-90, 91, 10), minor=True)
    ax3.set_yticks(np.arange(0.0, 0.21, 0.05))
    ax3.set_yticks(np.arange(0.0, 0.21, 0.01), minor=True)
    ax3.grid(alpha=0.2)
    ax4.set_xticks(range(-90, 91, 30))
    ax4.set_xticks(range(-90, 91, 10), minor=True)
    ax4.set_yticks(np.arange(0.0, 0.21, 0.05))
    ax4.set_yticks(np.arange(0.0, 0.21, 0.01), minor=True)
    ax4.grid(alpha=0.2)
    
    handles = [Line2D([0], [0], color=colors[j]) for j in range(nshell)]
    labels = [f"{o['name'].split(' ')[0]} ({o['number_of_planes'] * o['satellites_per_plane']} @ {o['altitude_km']:g} km / ${o['inclination_deg']:g}^\circ$)" for o in orbital_shells]

    fig.legend(handles, labels, ncol=3, loc="lower center", borderaxespad=0.1, bbox_to_anchor=(0.5, 1))
    plt.tight_layout()
    plt.savefig("density_with_latitude.png", bbox_inches="tight")
