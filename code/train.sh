#!/usr/bin/env bash

T=`date +%m%d%H%M`
CONFIG=$1

python main.py --config=$CONFIG \
    2>&1 | tee train.log.$T
