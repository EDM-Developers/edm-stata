# Empirical Dynamic Modeling (edm) Stata Package

## Package Description

The `edm` Stata package implements a series of  _Empirical Dynamic Modeling_ tools that can be used for causal analysis of time series data.

Key features of the package:

- powered by a fast multi-threaded _C++ backend_,
- able to process panel data, a.k.a. _multispatial EDM_,
- able to handle _missing data_ using new `dt` algorithms or by dropping points,
- _factor variables_ can be added to the analysis,
- _multiple distance functions_ available (Euclidean, Mean Absolute Error, Wasserstein),
- _GPU acceleration_ possible.
<!-- 
- so-called _coprediction_ is also available,
- forecasting methods will soon be added (WIP).
- training/testing splits can be made in a variety of ways including _cross-validation_,
-->

## R language version

We are currently creating the [fastEDM R package](https://edm-developers.github.io/fastEDM/) which is a direct port of this Stata package to R.
As both packages share the same underlying C++ code, their behaviour will be identical.

## Installation

To install the stable version directly through Stata:

``` stata
ssc install edm, replace
```

To install the latest development version, first install the stable version from SSC then inside Stata run:

``` stata
edm update, development replace
```

The source code for the package is available on [Github](https://github.com/EDM-Developers/EDM).

## Other Resources

This site serves as the primary source of documentation for the package, though there is also:

- our [Stata Journal paper](https://jinjingli.github.io/edm/edm-wp.pdf) which explains the package and the overall causal framework, and
- Jinjing's QMNET seminar on the package, the recording is on [YouTube](https://youtu.be/kZv85k1YUVE) and the [slides are here](pdfs/EDM-talk-QMNET.pdf).
