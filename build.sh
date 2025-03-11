#!/bin/bash
apt-get update
apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev libpng-dev libjpeg-dev libatlas-base-dev gfortran
pip install --verbose -r requirements.txt