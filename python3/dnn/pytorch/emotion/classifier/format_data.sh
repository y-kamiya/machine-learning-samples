#/bin/bash

src_path=$1

src_name=$(basename $src_path)

if [ $src_name == 'tweets_clean.txt' ]; then
    cat $src_path \
    | awk -F'\t' '{print($2, "\t", $3)}' \
    | sed -e 's/\t\s*::\s*/\t/' -e 's/"//g' \
    | sed -e 's/&lt;/</g' -e 's/&gt;/>/g' -e 's/&amp;/\&/g' -e "s/&quot;/'/g"

    exit
fi

if [ $src_name == 'text_emotion.csv' ]; then
    cat $src_path \
    | tail -n +2 \
    | python3 -c 'import csv, sys; csv.writer(sys.stdout, dialect="excel-tab").writerows(csv.reader(sys.stdin))' \
    | sed -e 's///g' -e 's/"//g' \
    | sed -e 's/&lt;/</g' -e 's/&gt;/>/g' -e 's/&amp;/\&/g' -e "s/&quot;/'/g" \
    | awk -F'\t' '{print $4"\t"$2}'

    exit
fi

src_dir_name=$(basename $(dirname $src_path))
if [ $src_dir_name == 'semeval2018' ]; then
    cat $src_path \
    | tail -n +2 \
    | sed -e 's///g' -e 's/"//g' \
    | sed -e 's/&lt;/</g' -e 's/&gt;/>/g' -e 's/&amp;/\&/g' -e "s/&quot;/'/g" \

    exit
fi
