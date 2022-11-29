# Empirical Dynamic Modelling (edm) Stata package

Please visit the package's [homepage](https://edm-developers.github.io/edm-stata) for a description of this plugin, documentation, and examples.

This repository stores the source code (Stata ado and C++) for the plugin.

Our dependencies include:
- the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) math library is used under Mozilla Public License 2, and
- [arrayfire][1] which can be downloaded from [ArrayFire download page][2].
- [CUDA 11.\*][3]

[1]: https://github.com/arrayfire/arrayfire
[2]: https://arrayfire.com/download
[3]: https://developer.nvidia.com/cuda-downloads

# Build Instructions

Given below is the CMake command to build the project using ArrayFire and CUDA.

```bash
mkdir build
cd build
cmake .. -DArrayFire_DIR:PATH="$AF_PATH/share/ArrayFire/cmake" -DCMAKE_TOOLCHAIN_FILE:FILEPATH="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
```
where

- `AF_PATH` variable points to ArrayFire installation root.
- `VCPKG_ROOT` variable  points to vcpkg repository root folder.

We also have provided a new CMake option `EDM_WITH_ARRAYFIRE` that compiles EDM with ArrayFire support.
When this option is set to OFF, the resulting edm binary can be used for systems where there are no CUDA GPUs.
