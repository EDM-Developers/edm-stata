cmake_minimum_required(VERSION 3.12)

# There are three targets in this CMake project:
# - edm_plugin makes the main plugin (DLL) which is loaded by Stata.
# - edm_cli creates a command-line executable to run the core EDM
#   algorithms by reading & saving data to JSON files. This doesn't
#   require Stata to run (unlike the main plugin).
# - gbench creates a Google Benchmarks executable which performs
#   microbenchmarking on the core EDM methods.

# Optionally, the ARMMODE environment variable specifies whether we are
# compiling for ARM or x86, though it defaults to x86 and can normally
# be ignored.
if(DEFINED ENV{ARMMODE})
  set(ARMMODE TRUE CACHE BOOL "Compiling to ARM")
else()
  set(ARMMODE FALSE CACHE BOOL "Compiling to x86")
endif()

# Optionally build the Google Benchmarks binary
if(DEFINED ENV{GBENCH})
  set(GBENCH TRUE CACHE BOOL "Building the Google Benchmarks target")
else()
  set(GBENCH FALSE CACHE BOOL "Building the Google Benchmarks target")
endif()

# The default behaviour of the 'install' target is to build everything first.
# As we often don't want to build the CLI executable, we disable this behaviour.
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

project(edm-stata-package CXX)

option(EDM_WITH_ARRAYFIRE "Enable ArrayFire based GPU support" OFF)

# Only include vcpkg for developer builds
if(DEFINED ENV{VCPKG_ROOT})
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Core requirements for all builds
if(NOT WIN32)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()
find_package(Threads REQUIRED)

# ↓↓↓ Third-party code ↓↓↓
if (EDM_WITH_ARRAYFIRE)
  find_package(CUDA REQUIRED)
  if (NOT WIN32)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  endif ()

  find_package(ArrayFire CONFIG REQUIRED)

  set(CUDA_architecture_build_targets "Pascal Volta Turing Ampere" CACHE
    STRING "The compute architectures targeted by this build. (Options: Auto;3.0;Maxwell;All;Common)")

  CUDA_SELECT_NVCC_ARCH_FLAGS(cuda_architecture_flags ${CUDA_architecture_build_targets})

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};${cuda_architecture_flags})
endif ()

# Output paths
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})

# Write compile_commands.json file (used by IDEs & static analysers)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Compiler flags
if(WIN32)
  set(SYSTEM "STWIN32" CACHE STRING "Stata plugin SYSTEM option")
else()
  set(SYSTEM "APPLEMAC" CACHE STRING "Stata plugin SYSTEM option")
endif()
set(CACHE SYSTEM PROPERTY STRINGS APPLEMAC HP9000 OPUNIX STWIN32)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYSTEM=${SYSTEM}")

if(WIN32)
  set(CMAKE_CXX_FLAGS_DEBUG "/MDd /W4 /permissive /ZI /fsanitize=address /bigobj")
  set(CMAKE_CXX_FLAGS_RELEASE "/MD /Ox /fp:fast /external:anglebrackets /external:W0")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
endif()

# Tell MSVC to compile in parallel
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)

# Source files common to all targets
set(COMMON_SRC src/cli.cpp src/cli.h
               src/common.cpp src/common.h
               src/cpu.cpp src/cpu.h
               src/distances.cpp src/distances.h
               src/edm.cpp src/edm.h
               src/manifold.cpp src/manifold.h
               src/stats.cpp src/stats.h
               src/library_prediction_split.h
               vendor/mersennetwister.h
               vendor/core.h vendor/EMD_wrapper.h
               vendor/network_simplex_simple.h vendor/full_bipartitegraph.h
               vendor/thread_pool.h
               )
if (EDM_WITH_ARRAYFIRE)
  if (WIN32)
    CUDA_ADD_LIBRARY(cu_lpDistance
      src/lp_distance.cu
      src/lp_distance.cuh
      STATIC)
  else ()
    CUDA_ADD_LIBRARY(cu_lpDistance
      src/lp_distance.cu
      src/lp_distance.cuh
      STATIC
      OPTIONS
      -Xcompiler -fPIC)
  endif ()
endif ()

# The main EDM plugin for Stata - minimal external dependencies
add_library(edm_plugin SHARED ${COMMON_SRC} src/main/stata.cpp vendor/stplugin.h)

# Rename the output plugin to 'edm_Windows_x64.plugin' or 'edm_MacOSX_arm.plugin' as appropriate
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

if (EDM_WITH_ARRAYFIRE)
  set(BUILD_SUFFIX "${BUILD_SUFFIX}_gpu")
endif()

set(BUILD_SUFFIX ${BUILD_SUFFIX} CACHE STRING "Compile to edm{BUILD_SUFFIX}.plugin")

if (ARMMODE)
  set_target_properties(edm_plugin
    PROPERTIES
      CXX_STANDARD 20
      PREFIX ""
      POSITION_INDEPENDENT_CODE ON
      OUTPUT_NAME "edm"
      OSX_ARCHITECTURES "arm64"
      SUFFIX ${BUILD_SUFFIX}.plugin
  )
else()
  set_target_properties(edm_plugin
    PROPERTIES
      CXX_STANDARD 20
      PREFIX ""
      POSITION_INDEPENDENT_CODE ON
      OUTPUT_NAME "edm"
      OSX_ARCHITECTURES "x86_64"
      SUFFIX ${BUILD_SUFFIX}.plugin
  )
endif()

# Core plugin dependencies
target_include_directories(edm_plugin PRIVATE src)
target_include_directories(edm_plugin SYSTEM PRIVATE 
    vendor 
    vendor/eigen-3.4.0 
    ${CMAKE_SOURCE_DIR}/vendor/json/include  # Path for vendored nlohmann/json
)
target_compile_definitions(edm_plugin PRIVATE JSON NOMINMAX)
target_link_libraries(edm_plugin PRIVATE Threads::Threads)

if (EDM_WITH_ARRAYFIRE)
  target_compile_definitions(edm_plugin PRIVATE WITH_ARRAYFIRE)
  target_include_directories(edm_plugin SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(edm_plugin PRIVATE ArrayFire::afcuda cu_lpDistance)
endif()

# Store output as both bin/edm_OS_ARCH.plugin and test/edm.plugin
install(TARGETS edm_plugin DESTINATION bin)
if (EDM_WITH_ARRAYFIRE)
  install(FILES bin/edm${BUILD_SUFFIX}.plugin DESTINATION test RENAME edm_gpu.plugin)
else()
  install(FILES bin/edm${BUILD_SUFFIX}.plugin DESTINATION test RENAME edm.plugin)
endif()
install(FILES stata/edm.ado DESTINATION test)

# To make CI builds easier to access, have the outputs be copied to Dropbox also.
# Copy to $EDM_CI_DIR, though if on the master branch, also copy to $EDM_CI_DIR-master.
set(CI_DIR $ENV{EDM_CI_DIR} CACHE STRING "Location of CI output directory")

if(DEFINED ENV{EDM_CI_DIR})
  # Get the current working branch
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  install(FILES bin/edm${BUILD_SUFFIX}.plugin stata/edm.ado DESTINATION $ENV{EDM_CI_DIR})
  if (${GIT_BRANCH} STREQUAL master)
    install(FILES bin/edm${BUILD_SUFFIX}.plugin stata/edm.ado DESTINATION $ENV{EDM_CI_DIR}-master)
  endif()
endif()

# Formatting with clang-format
include(ClangFormat)
clangformat_setup(${COMMON_SRC} src/main/benchmark.cpp src/main/cli_main.cpp src/main/test.cpp src/main/stata.cpp)

# ↓↓↓ Developer builds (require vcpkg) ↓↓↓
if(DEFINED ENV{VCPKG_ROOT})
  # Command-line interface executable
  add_executable(edm_cli ${COMMON_SRC} src/main/cli_main.cpp)

  set_target_properties(edm_cli
    PROPERTIES
      CXX_STANDARD 20
      POSITION_INDEPENDENT_CODE ON
      OSX_ARCHITECTURES "arm64;x86_64"
  )

  target_compile_definitions(edm_cli PRIVATE JSON)
  target_compile_definitions(edm_cli PRIVATE NOMINMAX)
  target_include_directories(edm_cli PRIVATE src)
  target_include_directories(edm_cli SYSTEM PRIVATE vendor vendor/eigen-3.4.0 vendor/json)
  target_link_libraries(edm_cli PRIVATE
    Threads::Threads
    )
  if (EDM_WITH_ARRAYFIRE)
    target_compile_definitions(edm_cli PRIVATE WITH_ARRAYFIRE)
    target_include_directories(edm_cli SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(edm_cli PRIVATE ArrayFire::afcuda cu_lpDistance)
  endif ()
  install(TARGETS edm_cli DESTINATION bin OPTIONAL)
  install(TARGETS edm_cli DESTINATION test OPTIONAL)

  # Google Benchmarks
  if(GBENCH)
    add_executable(gbench ${COMMON_SRC} src/main/benchmark.cpp)
    set_target_properties(gbench
      PROPERTIES
        CXX_STANDARD 20
        PREFIX ""
        POSITION_INDEPENDENT_CODE ON
    )

    # The google benchmarks currently won't build with
    # the GPU version due to some static/dynamic incompatibility.
    if (NOT (EDM_WITH_ARRAYFIRE AND WIN32))
      if(WIN32)
        target_compile_options(gbench PRIVATE "/MD")
      endif()

      find_package(benchmark CONFIG REQUIRED)
      target_compile_definitions(gbench PRIVATE JSON)
      target_compile_definitions(gbench PRIVATE NOMINMAX)
      target_link_libraries(gbench PRIVATE benchmark::benchmark benchmark::benchmark_main)
      target_include_directories(gbench PRIVATE src)
      target_include_directories(gbench SYSTEM PRIVATE vendor vendor/eigen-3.4.0 vendor/json)
      target_link_libraries(gbench PRIVATE Threads::Threads)
      if (EDM_WITH_ARRAYFIRE)
        target_compile_definitions(gbench PRIVATE WITH_ARRAYFIRE)
        target_include_directories(gbench SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(gbench PRIVATE ArrayFire::afcuda cu_lpDistance)
      endif ()
      install(TARGETS gbench DESTINATION test OPTIONAL)

    endif()
  endif()

  # Test executable (optional)
  find_package(Catch2 3 QUIET)
  if(Catch2_FOUND)
    add_executable(edm_test ${COMMON_SRC} src/main/test.cpp)
    set_target_properties(edm_test PROPERTIES CXX_STANDARD 20)
    target_compile_definitions(edm_test PRIVATE NOMINMAX)
    target_include_directories(edm_test PRIVATE src)
    target_include_directories(edm_test SYSTEM PRIVATE vendor vendor/eigen-3.4.0 vendor/json)
    if(WIN32)
      # Needs "/MDd" to compile in debug mode, however it still won't run in that mode..
      target_compile_options(edm_test PRIVATE "/MD")
    endif()

    target_link_libraries(edm_test PRIVATE Catch2::Catch2WithMain)
    target_link_libraries(edm_test PRIVATE Threads::Threads)
    if (EDM_WITH_ARRAYFIRE)
      target_compile_definitions(edm_test PRIVATE WITH_ARRAYFIRE)
      target_include_directories(edm_test SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
      target_link_libraries(edm_test PRIVATE ArrayFire::afcuda cu_lpDistance)
    endif ()

    install(TARGETS edm_test DESTINATION test OPTIONAL)
  endif()
endif()
