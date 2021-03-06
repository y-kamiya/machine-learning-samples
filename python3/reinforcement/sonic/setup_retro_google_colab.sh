#!/bin/bash

pip install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip install torchvision

# intall gym-retro
pip install gym-retro
pip install gym[atari]

# apt-get update -y
# apt-get install -y --allow-unauthenticated cmake libopenmpi-dev python3-dev zlib1g-dev
#
# git clone https://github.com/openai/baselines.git
# pushd baselines
# pip install -e .
# popd

dir=$(dirname $0)
cp $dir/scenario.json /usr/local/lib/python3.6/dist-packages/retro/data/stable/SonicTheHedgehog-Genesis/

# you should rom.md on current directory in advance
cp rom.md /usr/local/lib/python3.6/dist-packages/retro/data/stable/SonicTheHedgehog-Genesis/
