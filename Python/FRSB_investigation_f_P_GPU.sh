#!/bin/bash

#A script to investigate the behavior of f, P, here for a few values of alpha

#Values of my parameters
xmaxs=(15)
ks=(100 200 300)
cs=(30)
Hs=(40)
mlogecv=4 #Corresponds to 1e-mlogecv
alpha0=4
alpha1=8
NB=3 #alphas = linspace(alpha0, alpha1, num = NB, endpoint=True)
DEVICE="GPU"
NBGPU=0 #The number of the GPU to use

screen -S frsb_${DEVICE}_investigation_f_P -L -Logfile logs/frsb_${DEVICE}_investigation_f_P -dm zsh -c "python3 run_FRSB_$DEVICE.py --verbosity 1 --xmaxs ${xmaxs[*]} --Hs ${Hs[*]} --ks ${ks[*]} --cs ${cs[*]} --mlog_ecv $mlogecv --alphas $alpha0 $alpha1 $NB --device $NBGPU --full_output 1"