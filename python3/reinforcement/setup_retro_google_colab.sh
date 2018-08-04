#!/bin/bash

# install pytorch for cuda 8
pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip install torchvision

# intall gym-retro
pip install gym-retro

apt-get update
apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

git clone https://github.com/openai/baselines.git
pushd baselines
pip install -e .
popd

# you should rom.md on current directory in advance
cp rom.md /usr/local/lib/python3.6/dist-packages/retro/data/SonicTheHedgehog-Genesis/
