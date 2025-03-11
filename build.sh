#!/bin/bash
apt-get update
apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev libpng-dev libjpeg-dev
pip install -r requirements.txt