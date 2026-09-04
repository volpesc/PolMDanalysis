cmake_minimum_required(VERSION 3.16)
project(MDanalysis LANGUAGES CXX)

# ──────────────────────────────────────────────────────────────────────────────
#  Language standard
# ──────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to a Release build when none is specified (e.g. plain `cmake ..`).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
endif()

# ──────────────────────────────────────────────────────────────────────────────
#  Dependencies
# ──────────────────────────────────────────────────────────────────────────────
find_package(MPI REQUIRED)     # required (msd, pressure use MPI)
find_package(OpenMP)           # optional (fellipsoid is multithreaded if present)

# ──────────────────────────────────────────────────────────────────────────────
#  Executable
# ──────────────────────────────────────────────────────────────────────────────
add_executable(analysis analysis.cpp)

# analysis.cpp includes the header-only modules as "func/<name>.hpp",
# resolved relative to the project root.
target_include_directories(analysis PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(analysis PRIVATE MPI::MPI_CXX)

if(OpenMP_CXX_FOUND)
    target_link_libraries(analysis PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found - fellipsoid will run multithreaded (--threads)")
else()
    message(STATUS "OpenMP not found - building single-threaded (fellipsoid ignores --threads)")
endif()

# ──────────────────────────────────────────────────────────────────────────────
#  Warnings
# ──────────────────────────────────────────────────────────────────────────────
target_compile_options(analysis PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)

# ──────────────────────────────────────────────────────────────────────────────
#  Sanitizers in Debug builds
#    cmake .. -DCMAKE_BUILD_TYPE=Debug   →   AddressSanitizer + UBSan
# ──────────────────────────────────────────────────────────────────────────────
set(MDA_SANITIZERS -fsanitize=address,undefined -fno-omit-frame-pointer)
target_compile_options(analysis PRIVATE
    $<$<CONFIG:Debug>:-O0 -g ${MDA_SANITIZERS}>)
target_link_options(analysis PRIVATE
    $<$<CONFIG:Debug>:${MDA_SANITIZERS}>)
