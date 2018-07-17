#!/bin/bash

sudo apt-get install guake \
    build-essential \
    libboost-all-dev \
    libfreetype6-dev \
    libtbb-dev \
    libxmu-dev libxi-dev libgl1-mesa-dev dos2unix git wget libglew-dev \
    libgmp-dev libmpfr-dev \
    libcgal-dev \
    checkinstall \
    mpich \
    cmake cmake-qt-gui \
    gfortran \
    libopenblas-dev liblapack-dev libblas-dev libatlas-base-dev libarpack2-dev libarpack++2-dev libblas-dev libatlas-base-dev

# then cuda toolkit if applicable
