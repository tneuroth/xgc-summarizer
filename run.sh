#!/bin/bash

#arguments
    #mesh file path
    #bfield file path
    #particle data directory or filepath if single particle file mode
    #units.m file path
    #output directory (where to write the summary)
    #whether the mpi process reads a just it's chunk of the particles in the file
    #whether to run the program in situ (waiting for steps to be written from simulation)
    #whether there is one particle file (vs different files per time step)

mpirun -np 4\
    ./summarizer /media/tn0/data/datasets/CODAR/xgc.mesh.bp \
                 /media/tn0/data/datasets/CODAR/xgc.bfield.bp \
                 /media/tn0/data/datasets/CODAR/ \
                 /media/tn0/data/datasets/CODAR/units.m \
                 /media/tn0/data/datasets/ITER_SummaryMPI/ \
                 true \
                 false \
                 true

