# Empirical Dynamic Modeling Stata Package

## Package Description

<img src="assets/logo-lorenz.svg" align="right" height="200px" width="200px" alt="logo" />

_Empirical Dynamic Modeling (EDM)_ is a way to perform _causal analysis on time series data_. 
The `edm` Stata package implements a series of EDM tools, including the convergent cross-mapping algorithm. 

Key features of the package:

- powered by a fast multi-threaded _C++ backend_,
- able to process panel data, a.k.a. _multispatial EDM_,
- able to handle _missing data_ using new `dt` algorithms or by dropping points,
- _factor variables_ can be added to the analysis,
- _multiple distance functions_ available (Euclidean, Mean Absolute Error, Wasserstein),
- [_GPU acceleration_](/gpu) available.

<!-- 
- so-called _coprediction_ is also available,
- forecasting methods will soon be added (WIP).
- training/testing splits can be made in a variety of ways including _cross-validation_,
-->

## Installation

To install the stable version directly through Stata:

``` stata
ssc install edm, replace
```

To install the latest development version, first install the stable version from SSC then inside Stata run:

``` stata
edm update, development replace
```

The source code for the package is available on [Github](https://github.com/EDM-Developers/edm-stata).

## R & Python packages

We are currently creating the [fastEDM R package](https://edm-developers.github.io/fastEDM-r/) and the [fastEDM Python package](https://edm-developers.github.io/fastEDM-python/) which are direct ports of this Stata package to R & Python.
As all the packages share the same underlying C++ code, their behaviour will be identical.

## Other Resources

This site serves as the primary source of documentation for the package, though there is also:

- our [Stata Journal paper](https://jinjingli.github.io/edm/edm-wp.pdf) which explains the package and the overall causal framework, and
- Jinjing's QMNET seminar on the package, the recording is on [YouTube](https://youtu.be/kZv85k1YUVE) and the [slides are here](pdfs/EDM-talk-QMNET.pdf).

## Authors

- [Jinjing Li](https://www.jinjingli.com/) (author),
- [Michael Zyphur](https://business.uq.edu.au/profile/14074/michael-zyphur) (author),
- [Patrick Laub](https://pat-laub.github.io/) (author, maintainer),
- Edoardo Tescari (contributor),
- Simon Mutch (contributor),
- George Sugihara (originator)

## Citation

Jinjing Li, Michael J. Zyphur, George Sugihara, Patrick J. Laub (2021), _Beyond Linearity, Stability, and Equilibrium: The edm Package for Empirical Dynamic Modeling and Convergent Cross Mapping in Stata_, Stata Journal, 21(1), pp. 220-258

``` bibtex
@article{edm-stata,
  title={Beyond linearity, stability, and equilibrium: The edm package for empirical dynamic modeling and convergent cross-mapping in {S}tata},
  author={Li, Jinjing and Zyphur, Michael J and Sugihara, George and Laub, Patrick J},
  journal={The Stata Journal},
  volume={21},
  number={1},
  pages={220--258},
  year={2021},
}
```
