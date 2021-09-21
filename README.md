# Empirical Dynamic Modelling (edm) Stata package

Please visit the package's [homepage](https://jinjingli.github.io/edm/) for a description of this plugin, documentation, and examples.

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

Currently, `arrayfire` branch uses GPU by default. This can be changed to use CPU by setting the
`useAF` boolean flag to `false` in the `src/edm.cpp` file inside the function `edm_task`.
