#!/bin/bash

paste metadata_alpha.csv metadata_kana.csv \
| shuf \
| ggrep -v -P '\p{Han}' \
> shuf.csv

cat shuf.csv | head -n 160 | cut -f1 > metadata_alpha_val.csv
cat shuf.csv | head -n 160 | cut -f2 > metadata_kana_val.csv
cat shuf.csv | tail -n+161 | cut -f1 > metadata_alpha_train.csv
cat shuf.csv | tail -n+161 | cut -f2 > metadata_kana_train.csv
