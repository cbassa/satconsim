# satconsim - Satellite Constellation Simulator

## Introduction
_satconsim_ simulates the impact of large constellations of satellites on astronomical observations and allows astronomers to determine how their specific instruments may be impacted by different satellite constellations.

Using analytical descriptions of the location of a satellite within an orbital shell defined by its orbital altitude and inclination, _satconsim_ can simulate the number of satellites from a specified satellite constellation that are present in an observation of a certain field-of-view and exposure time, and taken from a certain location on Earth. For example, this figure shows the number of satellites for an LSST observation at Rubin Observatory.
!![Example all sky plot](https://raw.githubusercontent.com/cbassa/satconsim/master/examples/example_allsky_plot.png)

## Installation
Dependency requirements are handled using pip. You can install these requirements through

`pip install -r requirements`

Note that _satconsim_ requires `python 3`. To create animated gifs, the `convert` tool from *Image-Magick* is also required.
