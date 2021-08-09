#!/bin/bash

ROOT_PATH=$1

mkdir -p $ROOT_PATH/wavs
find $ROOT_PATH -name '*.wav' -type f | xargs -I@ cp @ $ROOT_PATH/wavs/
find $ROOT_PATH -name transcript_utf8.txt -type f | xargs cat | sed -e 's/:/|/' > $ROOT_PATH/metadata.csv

