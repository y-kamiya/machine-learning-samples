#/bin/bash

src_path=$1
count_eval=$2

TMP_FILE=$(mktemp)
trap "rm -rf $TMP_FILE" EXIT

src_dir=$(dirname $src_path)

cat $src_path | shuf > TMP_FILE
cat TMP_FILE | head -n $count_eval > $src_dir/eval.txt
cat TMP_FILE | tail -n +$count_eval > $src_dir/train.txt
