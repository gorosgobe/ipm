#!/usr/bin/env bash

# to be run from ipm/ directory
python3 -m venv venv
source venv/bin/activate
pip3 install numpy torch torchvision pytorch-ignite opencv-python pandas imageio matplotlib
pip3 uninstall -y torch
pip3 install torch