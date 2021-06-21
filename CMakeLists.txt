cmake_minimum_required(VERSION 3.12)

if(DEFINED ENV{VCPKG_TARGET_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
  set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_TARGET_TRIPLET}" CACHE STRING "")
endif()
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()
if(DEFINED ENV{ARMMODE})
  set(ARMMODE TRUE CACHE BOOL "Compiling to ARM")
else()
  set(ARMMODE FALSE CACHE BOOL "Compiling to x86")
endif()

set(DUMP_INPUT TRUE CACHE BOOL "Dump plugin input data to json")
set(BUILD_DRIVER FALSE CACHE BOOL "Generate command-line executable for the plugin")
set(BUILD_GBENCH FALSE CACHE BOOL "Generate google benchmark executable")

project(edm-mdap CXX)

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

if(WIN32)
  set(CMAKE_CXX_FLAGS_DEBUG "/MTd /W4 /permissive /ZI")
  set(CMAKE_CXX_FLAGS_RELEASE "/MT /Ox")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
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
set(COMMON_SRC src/cpu.cpp src/cpu.h src/edm.cpp src/edm.h src/manifold.cpp src/manifold.h
               src/ThreadPool.cpp src/ThreadPool.h)

# plugin build and installation
add_library(edm SHARED ${COMMON_SRC} src/stata.cpp vendor/stplugin.cpp vendor/stplugin.h)
set_target_properties(edm PROPERTIES CXX_STANDARD 17 PREFIX "")

if(WIN32)
  set(BUILD_SUFFIX "_Windows")
elseif(APPLE)
  set(BUILD_SUFFIX "_MacOSX")
else()
  set(BUILD_SUFFIX "_Unix")
endif()

if (ARMMODE)
  set(BUILD_SUFFIX "${BUILD_SUFFIX}_arm")
else()
  set(BUILD_SUFFIX "${BUILD_SUFFIX}_x64")
endif()

set(BUILD_SUFFIX ${BUILD_SUFFIX} CACHE STRING "Compile to edm{BUILD_SUFFIX}.plugin")
set_target_properties(edm PROPERTIES SUFFIX ${BUILD_SUFFIX}.plugin)

install(TARGETS edm DESTINATION .)

# grab vendored files
include(vendor/CMakeLists.txt)
target_include_directories(edm PRIVATE vendor)

# ↓↓↓ third-party code ↓↓↓
find_package(Eigen3 REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

if(NOT WIN32)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()
find_package(Threads REQUIRED)

target_include_directories(edm PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(edm PRIVATE fmt::fmt-header-only)
target_link_libraries(edm PRIVATE Threads::Threads)

# ↓↓↓ dev builds ↓↓↓
if (DUMP_INPUT OR BUILD_DRIVER OR BUILD_GBENCH)

  if (DUMP_INPUT)
    target_link_libraries(edm PRIVATE nlohmann_json::nlohmann_json)
    set_target_properties(edm PROPERTIES COMPILE_DEFINITIONS DUMP_INPUT=TRUE)
  endif()

  if (BUILD_DRIVER)
    add_executable(driver ${COMMON_SRC} src/driver.cpp src/driver.h)
    set_target_properties(driver PROPERTIES CXX_STANDARD 17)
    set_target_properties(driver PROPERTIES OSX_ARCHITECTURES "arm64;x86_64")
    target_include_directories(driver PRIVATE include)
    target_include_directories(driver PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(driver PRIVATE fmt::fmt-header-only)
    target_link_libraries(driver PRIVATE Threads::Threads)
    target_link_libraries(edm PRIVATE nlohmann_json::nlohmann_json)
    install(TARGETS driver DESTINATION .)
  endif()

  # Google Benchmarks
  if (BUILD_GBENCH)
    add_executable(gbench ${COMMON_SRC} src/benchmark.cpp src/driver.h)
    set_target_properties(gbench PROPERTIES CXX_STANDARD 17 PREFIX "")
    if(WIN32)
      target_compile_options(gbench PRIVATE "/MD")
    endif()
    find_package(benchmark CONFIG REQUIRED)
    target_link_libraries(gbench PRIVATE benchmark::benchmark benchmark::benchmark_main)
    target_include_directories(gbench PRIVATE include)
    target_include_directories(gbench PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(gbench PRIVATE fmt::fmt-header-only)
    target_link_libraries(gbench PRIVATE Threads::Threads)
    target_link_libraries(edm PRIVATE nlohmann_json::nlohmann_json)
    if(WIN32)
      find_package(OpenMP)
      target_link_libraries(gbench PRIVATE OpenMP::OpenMP_CXX)
    endif()
    install(TARGETS gbench DESTINATION .)
  endif()
endif()

# ↓↓↓ formatting with clang-format ↓↓↓
include(ClangFormat)
clangformat_setup(${COMMON_SRC} src/benchmark.cpp src/driver.cpp src/driver.h src/stata.cpp)