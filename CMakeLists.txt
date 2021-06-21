cmake_minimum_required(VERSION 3.12)

# We require the environment variable VCPKG_ROOT to be set before
# running cmake. This contains the path to the vcpkg install directory.

if(DEFINED ENV{VCPKG_TARGET_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
  set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_TARGET_TRIPLET}" CACHE STRING "")
endif()
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(edm-stata-package CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Output paths
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})

# Write compile_commands.json file (used by IDEs & static analysers)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Compiler flags
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

# Set the default build type if not set by user
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  endif()
endif()

# Source files common to all targets
set(COMMON_SRC src/cpu.cpp src/cpu.h src/edm.cpp src/edm.h src/manifold.cpp src/manifold.h
               src/thread_pool.cpp src/thread_pool.h)

# ↓↓↓ Third-party code ↓↓↓
find_package(Eigen3 REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

if(NOT WIN32)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()
find_package(Threads REQUIRED)

add_executable(edm_cli ${COMMON_SRC} src/cli.cpp src/cli.h)
set_target_properties(edm_cli PROPERTIES CXX_STANDARD 17)
target_include_directories(edm_cli PRIVATE include)
target_include_directories(edm_cli PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(edm_cli PRIVATE fmt::fmt-header-only)
target_link_libraries(edm_cli PRIVATE Threads::Threads)
target_link_libraries(edm_cli PRIVATE nlohmann_json::nlohmann_json)
install(TARGETS edm_cli DESTINATION bin)
install(TARGETS edm_cli DESTINATION test)