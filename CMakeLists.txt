set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required (VERSION 2.6)
project(XGC-Summary)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source/)
set(CMAKE_CXX_COMPILER mpicxx)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fopenmp -frounding-math -DZLIB -DBZIP2")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CUDA_COMPILER nvcc)
set(CUDA_NVCC_FLAGS "-std=c++11 -O4 -m 64 --ptxas-options=-v --compiler-options '-fPIC' -o libkdtree-cu.so --shared")

set(CMAKE_BUILD_TYPE Release)
set(VTKm_DIR "/home/tn0/repos/vtkm-build/lib/cmake/vtkm-1.2")

include_directories(PUBLIC /usr/include /usr/include/glib-2.0 /usr/lib/glib-2.0/include /usr/include/hdf5/serial/ /home/tn0/repos/vtkm-kdtree-generic/ ${SRC_DIR} /home/tn0/vtkm-build/include/)
link_directories(/usr/lib/x86_64-linux-gnu/hdf5/serial/ /usr/lib/x86_64-linux-gnu /usr/lib /usr/lib64 )

find_package(VTKm  REQUIRED)

# add_library(kdtree-cu ${SRC_DIR}/kdtree/ParticleMeshInterpolator2D.cu)
add_library(kdtree ${SRC_DIR}/kdtree/ParticleMeshInterpolator2D.cpp)

target_link_libraries(kdtree vtkm_cont)
# target_link_libraries(kdtree-cu vtkm_cont)

add_executable(summarizer ${SRC_DIR}/SummarizerMPIAdios1VTKmInterpolation.cpp)
# add_executable(bptest ${SRC_DIR}/Minimal.cpp)

target_link_libraries(summarizer vtkm_cont adios -lm -lmxml -lhdf5_hl -lhdf5 -lz -libverbs -lglib-2.0 -lz -lbz2 -Wl,-Bsymbolic-functions -Wl,-z,relro -lpthread CGAL gmp kdtree)
# target_link_libraries(bptest adios2 pthread mpi)
