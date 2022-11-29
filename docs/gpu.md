# GPU acceleration

## Setup

The `edm` commands normally run C++ code which run on multiple threads of a CPU, however it is possible to use the `gpu` option to move the heavy computation to an attached GPU.

To use this GPU acceleration, first make sure you are using a *Windows* or *Linux* machine which has an *NVIDIA graphics card* attached (preferably one of either Pascal, Volta, or Turing generations, though others may work if compiled locally).
Make sure your graphics drivers are installed (& relatively up-to-date), and *install* [*CUDA*](https://developer.nvidia.com/cuda-downloads) version 11 or above.
Next, *install* [*ArrayFire*](https://arrayfire.com/download/), making sure to select the 'Add to PATH' option.
Finally, make sure the latest development version of the `edm` package is installed by running

``` stata
ssc install edm // If no version of edm has been installed
edm update, development replace // To get the most up-to-date version
```

and your machine should be ready to run the GPU-accelerated.

## Usage

That means, any `edm explore` or `edm xmap` command can use the `gpu` option and the processing will use the attached GPU.

E.g.

``` stata
edm explore x, gpu
edm xmap x y, gpu
```

Alternatively, if you set the global `EDM_GPU = 1` then all subsequent `edm` commands will also run on the GPU, e.g.

``` stata
global EDM_GPU = 1
edm explore x
edm xmap x y
```

## Contributors

The GPU code was contributed by the ArrayFire engineers:

- Pradeep Garigipati,
- Umar Arshad,
- John Melonakos.

This collaboration was funded by the ARC Discovery Project DP200100219.