cmake_minimum_required(VERSION 3.1)

if(DEFINED ENV{VCPKG_TARGET_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
  set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_TARGET_TRIPLET}" CACHE STRING "")
endif()
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(edm-mdap CXX C)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# output paths
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/stata)

# write compile_commands.json file
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# compile flags
if(WIN32)
  set(SYSTEM "STWIN32" CACHE STRING "Stata plugin SYSTEM option")
else()
  set(SYSTEM "APPLEMAC" CACHE STRING "Stata plugin SYSTEM option")
endif()
set(CACHE SYSTEM PROPERTY STRINGS APPLEMAC HP9000 OPUNIX STWIN32)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYSTEM=${SYSTEM}")

set(DUMP_INPUT TRUE CACHE BOOL "Dump plugin input data to hdf5")

if(WIN32)
  set(CMAKE_CXX_FLAGS_DEBUG "/MTd /W4 /permissive")
  set(CMAKE_CXX_FLAGS_RELEASE "/MT")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
  set(REQUESTED_DEBUG_FLAGS "-fsanitize=address" "-fsanitize=leak" "-fsanitize=undefined" "-fno-common" "-fno-omit-frame-pointer")
endif()

include(CheckCXXCompilerFlag)

foreach(FLAG IN LISTS REQUESTED_DEBUG_FLAGS)
  string(REPLACE "-" "_" FLAG_NAME ${FLAG})
  string(REPLACE "=" "_" FLAG_NAME ${FLAG_NAME})
  check_cxx_compiler_flag(${FLAG} SUPPORTS_${FLAG_NAME})
  if(SUPPORTS_${FLAG_NAME})
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " ${FLAG}")
  endif()
endforeach()

# set the default build type if not set by user
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  endif()
endif()

# source list
set(SRC
    src/edm.cpp
    src/edm.hpp
    src/stata.cpp
    vendor/stplugin.cpp
    vendor/stplugin.hpp
    )

# plugin build and installation
add_library(edm SHARED ${SRC})
set_target_properties(edm PROPERTIES CXX_STANDARD 11 PREFIX "")

if(WIN32)
  set_target_properties(edm PROPERTIES SUFFIX _Windows_x64.plugin)
elseif(APPLE)
  set_target_properties(edm PROPERTIES SUFFIX _MacOSX_x64.plugin)
else()
  set_target_properties(edm PROPERTIES SUFFIX _Unix_x64.plugin)
endif()

install(TARGETS edm DESTINATION .)

# driver
find_package(HDF5 COMPONENTS C HL)
if (HDF5_FOUND)
  add_executable(smap_driver src/driver.cpp)
  set_target_properties(smap_driver PROPERTIES CXX_STANDARD 11)
  target_link_libraries(smap_driver PRIVATE edm)
  target_include_directories(smap_driver PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
  install(TARGETS smap_driver DESTINATION .)
endif()

# ↓↓↓ third-party code ↓↓↓

# grab vendored files
include(vendor/CMakeLists.txt)
target_include_directories(edm PRIVATE vendor)

# GSL
find_package(GSL REQUIRED)
target_link_libraries(edm PRIVATE GSL::gsl GSL::gslcblas)

# EIGEN3
find_package(EIGEN3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# OPENMP
find_package(OpenMP)
target_link_libraries(edm PRIVATE OpenMP::OpenMP_CXX)

# HDF5
if (HDF5_FOUND)
  target_include_directories(edm PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(edm PUBLIC ${HDF5_C_LIBRARIES} ${HDF5_C_HL_LIBRARIES})
  add_definitions(${HDF5_DEFINITIONS})
else()
  set(DUMP_INPUT FALSE)
  message(WARNING "HDF5 library not found. Unable to dump input values.")
endif()

# we can only assess this after we have checked if HDF5 is available
if(DUMP_INPUT)
  add_definitions(-DDUMP_INPUT)
endif()

# ↓↓↓ formatting with clang-format ↓↓↓
include(ClangFormat)
clangformat_setup(
  src/edm.cpp
  src/edm.hpp
  src/driver.cpp
  src/stata.cpp
)
