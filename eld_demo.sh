#!/usr/bin/env bash
source /home/minho/anaconda3/bin/activate demo
export FLASK_APP=src/demo/ELDAPI.py
export FLASK_ENV=development
python -m flask run --host=0.0.0.0 --port=6122 --no-reload
