#!/usr/bin/env bash
# Unused, now venv is in /vol/bitbucket
source venv/bin/activate
pip3 install numpy torch torchvision pytorch-ignite opencv-python pandas imageio matplotlib
pip3 uninstall -y torch
pip3 install torch