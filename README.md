# satconsim - Satellite Constellation Simulator

## Introduction
_satconsim_ simulates the impact of large constellations of satellites on astronomical observations and allows astronomers to determine how their specific instruments may be impacted by different satellite constellations.

Using analytical descriptions of the location of a satellite within an orbital shell defined by its orbital altitude and inclination, _satconsim_ can simulate the number of satellites from a specified satellite constellation that are present in an observation of a certain field-of-view and exposure time, and taken from a certain location on Earth. For example, this figure shows the number of satellites for an LSST observation at Rubin Observatory.

![Example all sky plot](https://raw.githubusercontent.com/cbassa/satconsim/master/examples/example_allsky_plot.png)

This software is the result of a collaboration between Cees Bassa (ASTRON), David Galadi (Calar Alto Observatory) and Olivier Hainaut (ESO) while preparing for the [SATCON1 Workshop](https://aas.org/satellite-constellations-1-workshop).

## Installation
Dependency requirements are handled using pip. You can install these requirements through

`pip install -r requirements`

Note that _satconsim_ requires `python 3`. To create animated gifs, the `convert` tool from *Image-Magick* is also required.

## Tools & Usage
* `nsat_allsky_anim.py`: Generates all sky maps with the number of satellites for a given constellation configuration and instrument configuration. It can generate a single plot (use `-n 1`) or, e.g. 20 plots at 5 minute intervals (`-n 20 -d 5`) at the provided start time (e.g. `-t 2020-07-14T18:00:00`). If the extention of the output is `gif`, the plots will be combined into an animated gif. The example plot above was created with `./nsat_allsky_anim.py -t 2020-06-20T23:00:00 -n 1 -o example_plot.png -c instruments/rubin_lsst.yaml -C constellations/starlink_gen2.yaml`.

## Satellite constellation configuration files
The satellite configuration is described in `yaml` files, using the following formatting to specify the constellation name and provide the parameters of each orbital shell in the constellation.

```yaml
name: "Starlink Generation 2"
shells:
- name: "Starlink Gen2 Shell 1"
  id: "starlink_gen2_shell_1"
  altitude_km: 328
  inclination_deg: 30.0
  number_of_planes: 7178
  satellites_per_plane: 1
- name: "Starlink Gen2 Shell 2"
  id: "starlink_gen2_shell_2"
  altitude_km: 334
  inclination_deg: 40.0
  number_of_planes: 7178
  satellites_per_plane: 1
```

## Instrument configuration files
The instrument configuration is also described in `yaml` files, using the following formatting.

```yaml
name: "Legacy Survey of Space and Time (LSST)"
site: "Rubin Observatory"
longitude_deg: -70.749417
latitude_deg: -30.244639
elevation_m: 2663
fov_deg_sq: 9.6
texp_s: 30
```
