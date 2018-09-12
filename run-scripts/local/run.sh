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
    #whether the particle files are given in append mode or not

DATA_DIR=/media/tn0/data/datasets/CODAR/

mpirun -np 4\
    ../../summarizer ${DATA_DIR}/adios2cfg.xml \
                 ${DATA_DIR}/xgc.mesh.bp \
                 ${DATA_DIR}/xgc.bfield.bp \
                 ${DATA_DIR}/restart_dir/ \
                 ${DATA_DIR}/units.m \
                 ${DATA_DIR}/summary/ \
                 true \
                 false \
                 false


