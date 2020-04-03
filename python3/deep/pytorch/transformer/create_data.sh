#!/bin/bash

name=$1
vocab_size=${2:-1000}

script_dir=$(cd `dirname $0`; pwd)
data_dir=$script_dir/data/$name

pushd $script_dir

cat $data_dir/train.orig.en $data_dir/train.orig.fr $data_dir/test.orig.en $data_dir/test.orig.fr > $data_dir/spm_input
./sentencepiece/build/src/spm_train --input=$data_dir/spm_input --model_prefix=$name --vocab_size=$vocab_size --pad_id=3

./sentencepiece/build/src/spm_encode --model=$data_dir/$name.model --output_format=id < $data_dir/train.orig.en > $data_dir/train.en
./sentencepiece/build/src/spm_encode --model=$data_dir/$name.model --output_format=id < $data_dir/train.orig.fr > $data_dir/train.fr
./sentencepiece/build/src/spm_encode --model=$data_dir/$name.model --output_format=id < $data_dir/test.orig.en > $data_dir/test.en
./sentencepiece/build/src/spm_encode --model=$data_dir/$name.model --output_format=id < $data_dir/test.orig.fr > $data_dir/test.fr

mv $script_dir/$name.model $script_dir/$name.vocab $data_dir/

popd

