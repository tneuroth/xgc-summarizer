#!/bin/bash

mpirun -np 4\
    ./summarizer /home/tn0/xgc.mesh.bp \
                 /home/tn0/xgc.bfield.bp \
                 /home/tn0/ \
/home/tn0/units.m /media/tn0/data/datasets/ITER_SummaryMPI/
