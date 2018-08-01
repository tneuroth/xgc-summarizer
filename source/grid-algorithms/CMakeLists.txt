set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required (VERSION 2.6)
project(Decimator)

set(DEV ${CMAKE_CURRENT_SOURCE_DIR}/../../)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)

set(CMAKE_CXX_COMPILER g++)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fopenmp -frounding-math -D_NOMPI -DZLIB -DBZIP2")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS "-lm")

set(CMAKE_BUILD_TYPE Release)

include_directories(PUBLIC /home/tn0/repos/CGAL-Build/include /usr/include /usr/include/glib-2.0 /usr/lib/glib-2.0/include/ ${DEV}/adios-build/include/)
link_directories(/home/tn0/repos/CGAL-Build/lib /usr/lib/x86_64-linux-gnu /usr/lib /usr/lib64 ${DEV}/adios-build/lib)

add_executable(decimator ${SRC_DIR}/Decimate.cpp)
target_link_libraries(decimator adios_nompi z bz2 CGAL gmp)

