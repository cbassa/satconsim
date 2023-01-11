#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from satconsim.utils import azel

def plot_map(fig, ax, w, rx, ry, v, vmin, vmax, cmap, label, normtype="linear"):
    c = np.isfinite(v)
    if np.sum(c) > 0:
        #vmin, vmax = np.nanmin(v), np.nanmax(v)
        if normtype == "log":
            c = v > 0
            #vmin = np.nanmin(v[c])
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            
        vlevels = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        c = (vlevels > vmin) & (vlevels < vmax)

        img = ax.imshow(v, origin="lower", interpolation=None, aspect=1,
                        cmap=cmap, norm=norm)
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("bottom", size="5%", pad="8%")
        cbar = fig.colorbar(img, cax=cax, orientation="horizontal", label=label, ticks=vlevels, norm=norm)
        #cbar = mpl.colorbar.ColorbarBase(ax, cmap=mpl.cm.plasma, label=label,
        #                                 norm=norm, ticks=vlevels,
        #                                 orientation="horizontal")
        
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

def plot_sources(w, ax, lat, lst):
    alphamin = 0.5
    # Dec lines
    ra = np.arange(361)
    ha = np.mod(lst - ra, 360)
    for dec in [-60, -30, 0, 30, 60]:
        az, el = azel(ha, dec * np.ones_like(ha), lat)

        # Convert
        rx, ry = w.wcs_world2pix(np.stack((az, el), axis=-1), 1).T

        if dec == 0.0:
            color = "gray"
            linestyle="--"
            alpha = alphamin
        else:
            color = "gray"
            linestyle=":"
            alpha = 1.0
        c = el >= 0
        rx = np.ma.array(rx, mask=~c)
        ry = np.ma.array(ry, mask=~c)
        ax.plot(rx, ry, color, linestyle=linestyle, alpha=alpha)

    # RA lines
    dec = np.linspace(-90, 90, 100)
    for ra in [0, 45, 90, 135, 180, 225, 270, 315]:
        ha = np.mod(lst - ra, 360) * np.ones_like(dec)
        az, el = azel(ha, dec, lat)
        
        # Convert
        rx, ry = w.wcs_world2pix(np.stack((az, el), axis=-1), 1).T

        if ra == 0:
            color = "gray"
            linestyle="--"
            alpha = alphamin
        else:
            color = "gray"
            linestyle=":"
            alpha = 1.0        
        c = el >= 0
        rx = np.ma.array(rx, mask=~c)
        ry = np.ma.array(ry, mask=~c)
        ax.plot(rx, ry, color, alpha=alphamin, linestyle=linestyle)

    # SMC/LMC
    ra = np.array([13.1867, 80.8938])
    dec = np.array([-72.8286, -69.7561])
    ha = np.mod(lst - ra, 360)
    az, el = azel(ha, dec, lat)
    rx, ry = w.wcs_world2pix(np.stack((az, el), axis=-1), 1).T
    c = el >= 0
    ax.plot(rx[c], ry[c], marker="o", color="silver", linestyle="None")
    
    # Galactic plane
    ra = np.array([266.40498829, 269.26862916, 271.94042232, 274.46552058,
                      276.88229207, 279.22424989, 281.52164248, 283.802811  ,
                      286.09540955, 288.42757587, 290.82913688, 293.33293579,
                      295.97637495, 298.80327887, 301.86618803, 305.22917778,
                      308.97121783, 313.18985085, 318.00438151, 323.55647134,
                      330.00353904, 337.49657947, 346.1312935 , 355.8685336 ,
                      6.45083114,  17.38741142,  28.07149678,  37.98010947,
                      46.81328141,  54.49933533,  61.11706742,  66.81231157,
                      71.74356338,  76.05598318,  79.87281983,  83.29518346,
                      86.40498829,  89.26862916,  91.94042232,  94.46552058,
                      96.88229207,  99.22424989, 101.52164248, 103.802811  ,
                      106.09540955, 108.42757587, 110.82913688, 113.33293579,
                      115.97637495, 118.80327887, 121.86618803, 125.22917778,
                      128.97121783, 133.18985085, 138.00438151, 143.55647134,
                      150.00353904, 157.49657947, 166.1312935 , 175.8685336 ,
                      186.45083114, 197.38741142, 208.07149678, 217.98010947,
                      226.81328141, 234.49933533, 241.11706742, 246.81231157,
                      251.74356338, 256.05598318, 259.87281983, 263.29518346,
                      266.40498829])
    dec = np.array([-28.93617776, -24.63840463, -20.2900295 , -15.90335762,
                       -11.48861973,  -7.05460278,  -2.60914152,   1.84047872,
                       6.28716931,  10.72370325,  15.14236764,  19.53457212,
                       23.89037583,  28.197885  ,  32.44245686,  36.60562236,
                       40.6636104 ,  44.58532703,  48.32963721,  51.8418792 ,
                       55.04986526,  57.86047241,  60.15961986,  61.82060157,
                       62.72572675,  62.79899813,  62.0338429 ,  60.49575721,
                       58.29770514,  55.56678771,  52.42018612,  48.95480168,
                       45.24624231,  41.35190742,  37.31499947,  33.16813152,
                       28.93617776,  24.63840463,  20.2900295 ,  15.90335762,
                       11.48861973,   7.05460278,   2.60914152,  -1.84047872,
                       -6.28716931, -10.72370325, -15.14236764, -19.53457212,
                       -23.89037583, -28.197885  , -32.44245686, -36.60562236,
                       -40.6636104 , -44.58532703, -48.32963721, -51.8418792 ,
                       -55.04986526, -57.86047241, -60.15961986, -61.82060157,
                       -62.72572675, -62.79899813, -62.0338429 , -60.49575721,
                       -58.29770514, -55.56678771, -52.42018612, -48.95480168,
                       -45.24624231, -41.35190742, -37.31499947, -33.16813152,
                       -28.93617776])
    ha = np.mod(lst - ra, 360)
    az, el = azel(ha, dec, lat)
    rx, ry = w.wcs_world2pix(np.stack((az, el), axis=-1), 1).T
    c = el >= 0
    rx = np.ma.array(rx, mask=~c)
    ry = np.ma.array(ry, mask=~c)
    ax.plot(rx, ry, marker="None", color="silver", linestyle=":")
    if el[0] > 0:
        ax.plot(rx[0], ry[0], marker="o", color="silver", linestyle="None")
        

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

    return
