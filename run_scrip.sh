#!/bin/bash
nohup python -u lightgbm_797.py > lightgbm_797.log &
xnohup python -u xgboost_tune.py > xgboost_tune.log &
