#/bin/bash

src_path=$1

cat $src_path \
| awk -F'\t' '{print($2, "\t", $3)}' \
| sed -e 's/\t\s*::\s*/\t/' 
