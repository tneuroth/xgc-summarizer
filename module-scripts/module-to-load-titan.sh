module unload PrgEnv-pgi PrgEnv-gnu PrgEnv-cray
module load PrgEnv-gnu
#module swap gcc gcc/6.3.0
#module load cray-petsc/3.6.3.0
module load cray-petsc/3.7.6.1
module load cmake3/3.9.0
module load craype-hugepages2M
module load cudatoolkit
module load python python_numpy

module use -a /lustre/atlas/world-shared/csc143/jyc/titan/sw/modulefiles 
module load adios2/devel-tyson

export XGC_PLATFORM=titan.gcc.jyc 

