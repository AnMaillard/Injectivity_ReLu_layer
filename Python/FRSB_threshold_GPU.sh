#!/bin/bash

#A script to compute the injectivity threshold FRSB prediction for different values of parameters

xmaxs=(10 11 12 13 14 15)
ks=(30 50 100 150 200)
cs=(30)
Hs=(40 60)
mlogecv=5 #Corresponds to 1e-mlogecv
DEVICE="GPU"
NBGPU=1 #The number of the GPU to use

screen -S frsb_${DEVICE}_threshold -L -Logfile logs/frsb_${DEVICE}_threshold -dm zsh -c "python3 run_FRSB_$DEVICE.py --verbosity 1 --xmaxs ${xmaxs[*]} --Hs ${Hs[*]} --ks ${ks[*]} --cs ${cs[*]} --mlog_ecv $mlogecv --device $NBGPU"
