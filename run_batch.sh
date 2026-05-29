#!/bin/bash
# Wrapper to run the full dataset generation with proper LD_LIBRARY_PATH
source "$(dirname "$0")/venv_ml/bin/activate"
export LD_LIBRARY_PATH="$(dirname "$0")/venv_ml/lib/python3.11/site-packages/openseespylinux/lib:$LD_LIBRARY_PATH"
python "$(dirname "$0")/run_full_dataset.py" "$@"
