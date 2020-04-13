#!/bin/bash

curl -O https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml

if [ ! -d venv-object-detection ]; then
    python3 -m venv venv-object-detection
fi
source venv-object-detection/bin/activate

pip install opencv-python pascal-voc-writer
