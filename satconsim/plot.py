#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

def plot_map(fig, ax, w, rx, ry, v, cmap, label, normtype="linear"):
    c = np.isfinite(v)
    if np.sum(c) > 0:
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if normtype == "log":
            c = v > 0
            vmin = np.nanmin(v[c])
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            
        vlevels = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        c = (vlevels > vmin) & (vlevels < vmax)
        
        img = ax.imshow(v, origin="lower", interpolation=None, aspect=1,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        norm=norm)
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("bottom", size="5%", pad="8%")
        cbar = fig.colorbar(img, cax=cax, orientation="horizontal", label=label, ticks=vlevels)
        labels = []
        for vlevel in vlevels:
            if vlevel >= 1e-4:
                labels.append(f"${vlevel:g}$")
            else:
                vexp = int(np.floor(np.log10(vlevel)))
                vman = vlevel / 10**vexp
                labels.append(f"${vman:g}\\times 10^{{{vexp}}}$")
                
        cbar.ax.set_xticklabels(labels)
        
    ax.axis("off")
    plot_grid(w, ax)

    
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
