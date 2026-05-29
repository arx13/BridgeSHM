#!/bin/bash
# Activate the ML virtual environment with OpenSeesPy BLAS path
source /home/spctr/bridge_SHM/venv_ml/bin/activate
export LD_LIBRARY_PATH="/home/spctr/bridge_SHM/venv_ml/lib/python3.11/site-packages/openseespylinux/lib:$LD_LIBRARY_PATH"
exec "$@"
