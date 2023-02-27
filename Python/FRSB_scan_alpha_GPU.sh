#!/bin/bash

#A script to compute the FRSB prediction for different values of parameters, each time over a range of alpha

#Values of my parameters
xmaxs=(10 15)
ks=(30 50 100 200)
cs=(30)
Hs=(40)
mlogecv=4 #Corresponds to 1e-mlogecv
alpha0=3
alpha1=10
NB=50 #alphas = linspace(alpha0, alpha1, num = NB)
DEVICE="GPU"
NBGPU=0 #The number of the GPU to use

screen -S frsb_${DEVICE}_scan_alpha -L -Logfile logs/frsb_${DEVICE}_scan_alpha -dm zsh -c "python3 run_FRSB_$DEVICE.py --verbosity 1 --xmaxs ${xmaxs[*]} --Hs ${Hs[*]} --ks ${ks[*]} --cs ${cs[*]} --mlog_ecv $mlogecv --alphas $alpha0 $alpha1 $NB --device $NBGPU"