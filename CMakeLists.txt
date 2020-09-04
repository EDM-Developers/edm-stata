cmake_minimum_required(VERSION 3.1)
project(edm-mdap)

# output paths
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/stata)

# write compile_commands.json file
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# compile flags
set(SYSTEM "APPLEMAC" CACHE STRING "Stata plugin SYSTEM option")
set(CACHE SYSTEM PROPERTY STRINGS APPLEMAC HP9000 OPUNIX STWIN32)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSYSTEM=${SYSTEM}")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

include(CheckCCompilerFlag)
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(REQUESTED_DEBUG_FLAGS "-fsanitize=address" "-fsanitize=leak" "-fsanitize=undefined" "-fno-common" "-fno-omit-frame-pointer")
foreach(FLAG IN LISTS REQUESTED_DEBUG_FLAGS)
    string(REPLACE "-" "_" FLAG_NAME ${FLAG})
    string(REPLACE "=" "_" FLAG_NAME ${FLAG_NAME})
    check_c_compiler_flag(${FLAG} SUPPORTS_${FLAG_NAME})
    if(SUPPORTS_${FLAG_NAME})
        string(APPEND CMAKE_C_FLAGS_DEBUG " ${FLAG}")
    endif()
endforeach()

# set the default build type if not set by user
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
  endif()
endif()

# source list
set(SRC
    src/smap_block_mdap.c
    vendor/stplugin.c
    )

# plugin build and installation
add_library(smap_block_mdap MODULE ${SRC})
set_target_properties(smap_block_mdap PROPERTIES C_STANDARD 99 SUFFIX .plugin PREFIX "")
install(TARGETS smap_block_mdap DESTINATION .)

# ↓↓↓ third-party code ↓↓↓

# grab vendored files
include(vendor/CMakeLists.txt)
target_include_directories(smap_block_mdap PRIVATE vendor)

# GSL
find_package(GSL REQUIRED)
target_link_libraries(smap_block_mdap PRIVATE GSL::gsl GSL::gslcblas)

# OPENMP
find_package(OpenMP)
target_link_libraries(smap_block_mdap PRIVATE OpenMP::OpenMP_C)
