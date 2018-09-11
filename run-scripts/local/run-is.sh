#!/bin/bash

#arguments
    #adios2 config file
    #mesh file path
    #bfield file path
    #particle data directory or filepath if single particle file mode
    #units.m file path
    #output directory (where to write the summary)
    #whether the mpi process reads a just it's chunk of the particles in the file
    #whether to run the program in situ (waiting for steps to be written from simulation)
    #whether there is one particle file (vs different files per time step)

DATA_DIR=/media/tn0/data/datasets/inSituTest

mpirun -np 8\
    ../../summarizer ${DATA_DIR}/adios2cfg.xml \
                 ${DATA_DIR}/xgc.mesh.bp \
                 ${DATA_DIR}/xgc.bfield.bp \
                 ${DATA_DIR}/xgc.particle.bp \
                 ${DATA_DIR}/units.m \
                 ${DATA_DIR}/summary/ \
                 true \
                 true \
                 true

