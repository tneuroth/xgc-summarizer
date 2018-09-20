#!/bin/bash

#arguments
    #adios2 config file
    #mesh file path
    #bfield file path
    #particle data directory or filepath if single particle file mode
    #units.m file path
    #output directory (where to write the summary)
    #whether to split by blocks or just evenly split particles
    #whether to run the program in situ (waiting for steps to be written from simulation)
    #whether to the particle files were written in append mode or not
    #whether to try using cuda

DATA_DIR=/media/tn0/data/datasets/inSituTest

mpirun -np 8\
    ../../summarizer ${DATA_DIR}/adios2cfg.xml \
                 ${DATA_DIR}/xgc.mesh.bp \
                 ${DATA_DIR}/xgc.bfield.bp \
                 ${DATA_DIR}/xgc.particle.bp \
                 ${DATA_DIR}/units.m \
                 ${DATA_DIR}/summary/ \
                 false \
                 true \
                 true \
                 true

