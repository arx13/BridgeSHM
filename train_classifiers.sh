#!/bin/bash
source "$(dirname "$0")/venv_ml/bin/activate"
cd "$(dirname "$0")"
python models/train_classifier.py 2>&1 | tee outputs/classifiers/training.log
