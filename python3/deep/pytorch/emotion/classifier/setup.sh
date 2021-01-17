#!/bin/bash

USE_CPU=$1

git submodule update --init

pip install torch torchvision tensorboard logzero transformers

pushd apex
git checkout 11faaca7c8ff7a7ba6d55854a9ee2689784f7ca5
if $USE_CPU; then
    pip install -v --no-cache-dir --global-option="--cpp_ext" ./
else
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
fi
popd
