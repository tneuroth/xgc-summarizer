#!/bin/bash

mpirun -np 2\
    ./summarizer /media/tn0/data/datasets/XGC/ITER/Grid/xgc.mesh.h5 \
                 /media/tn0/data/datasets/XGC/ITER/Grid/xgc.bfield.h5 \
                 /media/tn0/data/datasets/XGC/ITER/ParticleData/ \
                 /media/tn0/data/datasets/XGC/ITER/units.m /media/tn0/data/datasets/ITER_SummaryMPI/

