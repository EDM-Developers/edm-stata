# Empirical Dynamic Modeling (edm) Stata Package

## Package Description

`edm` implements a series of tools that can be used for _Empirical Dynamic Modeling_ in Stata.

Key features of the package:

- fast (written in C++) and multithreaded,
- able to handle missing data,
- able to process panel data,
- multiple distance functions available (Euclidean, Mean Absolute Error, Wasserstein)
- GPU acceleration

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

## Resources

- This site serves as the primary source of documentation for the package.
- Our [Stata Journal paper](https://jinjingli.github.io/edm/edm-wp.pdf) is also a good starting place to understand the package and the overall causal framework.
- A QMNET seminar on the package is another good starting place, the recording of which ison [YouTube](https://youtu.be/kZv85k1YUVE) and the [slides are here](pdfs/EDM-talk-QMNET.pdf).
