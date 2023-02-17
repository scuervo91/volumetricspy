# Release Notes

# 0.1.10
### Fix
* Fix dependencies issues
# 0.1.9
### Fix
* Fix dependencies issues
# 0.1.8
### Features

* Create constants surfaces classmethod. This function helps to create surfaces like WOC
* Add add, sub, mul, trudiv operators to surfaces. now you can make operations between surfaces. The operations will take place in the Z attribute
* Add ability to sync the surfaces to have the same grid. To do this a Regular Grid Interpolator is used.
* Surfaces Difference 
* Integrate along surface. Usefull when calculating Volumetrics.

# 0.1.7
### Features

* Fix indexing mesh from 'ij' to 'xy'
* Add feature to estimate the volume between surfaces by interpolating a regular mesh
* Add feature to export surfaces as zmap format
# 0.1.6
### Features
* Add Inverse Weight Distance algorithm
# 0.1.5
### Fixes
👷 add mapclassify

# 0.1.4
### Fixes
👷 Update Geopandas version dependency to implement 'explore' method.

## 0.1.3
### Feature
* Add Krigging classes. Oridinary Krigging and Indicator Krigging
* Add Sequential Gaussian Simulation Class

### Fixes
* General Bug Fixes


## 0.1.2
### Feature
* 👷 Fix bugs when meshing

## 0.1.0
### Initial Release
* 👷 Fix bug when printing
Oil & Gas tool to estimate Original Resources in Place (OOIP OGIP) from a group of Surfaces. 

+ You can estimate Resources volumes probabilistically by applying MonteCarlo Simulation using Scipy built in probabilistic distributions.
+ Plot Contours maps Using Matplotlib
+ Show Locations map using Folium
+ Make 3D visualization Using PyVista
+ Make Montecarlo Simulation for resources