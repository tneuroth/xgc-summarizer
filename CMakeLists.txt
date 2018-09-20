set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required (VERSION 2.6)
project( XGC-Summary LANGUAGES CXX CUDA )

include(ParallelUtils.cmake)
enable_cuda_support()

option(TITAN OFF)
option(CORI OFF)
option(UBUNTU ON)
option(VTKM_CUDA OFF)

set(DEV ${CMAKE_CURRENT_SOURCE_DIR}/../)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source/)
set(VTKm_DIR "${DEV}/kdtree/vtkm-build/lib/cmake/vtkm-1.2")
  
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

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-w -fopenmp -frounding-math")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS "-lm")

set(CUDA_NVCC_FLAGS "-std=c++11 -O4 -Xcompiler \"\ -openmp\"")

set(CMAKE_BUILD_TYPE Release)

include_directories(
    PUBLIC
    /usr/include/
    /usr/include/glib-2.0/
    /usr/lib/glib-2.0/include/
    ${SRC_DIR}/
    ${DEV}/kdtree/vtkm-build/include/
    ${DEV}/adios-builds/include/
    ${ADIOS1_DIR}/include/
    ${ADIOS2_DIR}/include/
    ${DEV}/kdtree/vtkm-kdtree-generic/)

link_directories(
    /usr/lib/x86_64-linux-gnu/    
    ${DEV}/kdtree/vtkm-build/lib/    
    ${ADIOS1_DIR}/lib/
    ${ADIOS2_DIR}/lib/
    /usr/lib/
    /usr/lib64/)

find_package(VTKm REQUIRED)
find_package(CUDA REQUIRED)  

add_library(aggregator ${SRC_DIR}/XGCAggregator.cxx)
add_library(aggregatorTools ${SRC_DIR}/VTKmAggregatorTools.cu)

target_link_libraries( aggregatorTools vtkm_cont )

if( VTKM_CUDA ) 
    # set_target_properties( aggregatorTools PROPERTIES CUDA_SEPARABLE_COMPILATION ON ) 
    # set_target_properties( vtkm_cont PROPERTIES CUDA_SEPARABLE_COMPILATION ON )          
    # get_source_file_property( _cuda_source_format VTKmAggregatorTools.cxx CUDA_SOURCE_PROPERTY_FORMAT)
    # set_source_files_properties( VTKmAggregatorTools.cxx PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT _cuda_source_format )
endif()

if(TITAN)
    target_link_libraries(aggregator aggregatorTools adios2 adios2_sst pthread adios2_evpath 
        adios adios2_enet adios2_atl adios2_atl adios2_ffs adios2_dill vtkm_cont bz2 gmp)
    add_executable(summarizer ${SRC_DIR}/Main.cpp)
    target_link_libraries(summarizer adios aggregator)
elseif(CORI)
    target_link_libraries(aggregator aggregatorTools adios2 adios gmp vtkm_cont )
    add_executable(summarizer ${SRC_DIR}/Main.cpp)
    target_link_libraries(summarizer adios aggregator)
else()
    target_link_libraries(aggregator aggregatorTools adios2 bz2 pthread adios gmp vtkm_cont )
    add_executable(summarizer ${SRC_DIR}/Main.cpp)
    target_link_libraries(summarizer adios pthread bz2 aggregator)
endif()
