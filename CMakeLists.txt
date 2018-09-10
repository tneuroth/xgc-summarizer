set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required (VERSION 2.6)
project(XGC-Summary)

option(TITAN OFF)
option(CORI OFF)
option(UBUNTU ON)
option(VTKM_CUDA OFF)

set(DEV ${CMAKE_CURRENT_SOURCE_DIR}/../)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source/)
set(VTKm_DIR "${DEV}vtkm-build/lib/cmake/vtkm-1.2")
  
if(TITAN)
    set(CMAKE_C_COMPILER cc)
    set(CMAKE_CXX_COMPILER CC)
    set(ADIOS2_DIR "/lustre/atlas/world-shared/csc143/jyc/titan/sw/adios2/devel/gnu")
    set(ADIOS1_DIR "/lustre/atlas/world-shared/csc143/jyc/titan/sw/adios-matlab/devel/gnu") 
elseif(CORI)
    set(CMAKE_C_COMPILER cc)
    set(CMAKE_CXX_COMPILER CC)
    #set(ADIOS2_DIR "/global/cscratch1/sd/jyc/sw/cori.haswell.gnu/adios2/devel")     
    set(ADIOS2_DIR "${DEV}/adios2-builds/mpi/")  
    set(ADIOS1_DIR "${DEV}/adios-builds/")        
elseif (UBUNTU)
    set(CMAKE_CXX_COMPILER mpicxx)
    set(ADIOS2_DIR "${DEV}/adios2-builds/mpi/")  
    set(ADIOS1_DIR "${DEV}/adios-builds/")     
endif()

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-w -fopenmp -frounding-math")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS "-lm")
set(CMAKE_CUDA_COMPILER nvcc)
set(CUDA_NVCC_FLAGS "-std=c++11 -O3 -m 64 -Xcompiler \"\
-openmp\" --ptxas-options=-v --compiler-options '-fPIC'\
 -o libkdtree-cu.so --shared")

if( VTKM_CUDA )
    get_source_file_property( _cuda_source_format XGCAggregator.cxx CUDA_SOURCE_PROPERTY_FORMAT)
    set_source_files_properties( XGCAggregator.cxx PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT _cuda_source_format )
endif()

set(CMAKE_BUILD_TYPE Release)

include_directories(
    PUBLIC
    /usr/include/
    /usr/include/glib-2.0/
    /usr/lib/glib-2.0/include/
    ${DEV}/vtkm-kdtree-generic/
    ${SRC_DIR}/
    ${DEV}/vtkm-build/include/
    ${DEV}/adios-builds/include/
    ${ADIOS1_DIR|/include/
    ${ADIOS2_DIR}/include/
    ${DEV}/vtkm-kdtree-generic/)

link_directories(
    /usr/lib/x86_64-linux-gnu/    
    ${ADIOS1_DIR}/lib/
    ${ADIOS2_DIR}/lib/
    /usr/lib/
    /usr/lib64/)
    
find_package(VTKm REQUIRED)

add_library(kdtree ${SRC_DIR}/XGCAggregator.cxx)

if(TITAN)
    target_link_libraries(kdtree adios2 vtkm_cont)
    add_executable(summarizer ${SRC_DIR}/Main.cpp)
    target_link_libraries(summarizer vtkm_cont 
        adios2 adios2_sst pthread adios2_evpath adios adios2_enet adios2_atl adios2_atl adios2_ffs adios2_dill
        bz2 gmp kdtree)
elseif(CORI)
    target_link_libraries(kdtree adios2 vtkm_cont)
    add_executable(summarizer ${SRC_DIR}/Main.cpp)
    target_link_libraries(summarizer vtkm_cont 
        adios2 adios
        gmp kdtree)
else()
    target_link_libraries(kdtree adios2 vtkm_cont)
    add_executable(summarizer ${SRC_DIR}/Main.cpp)
    target_link_libraries(summarizer vtkm_cont adios2 adios gmp kdtree)
endif()
