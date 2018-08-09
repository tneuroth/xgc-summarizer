#!/bin/bash

mpirun -np 4\
    ./summarizer /media/tn0/data/datasets/CODAR/xgc.mesh.bp \
                 /media/tn0/data/datasets/CODAR/xgc.bfield.bp \
                 /media/tn0/data/datasets/CODAR/ \
                 /media/tn0/data/datasets/CODAR/units.m \
                 /media/tn0/data/datasets/ITER_SummaryMPI/
