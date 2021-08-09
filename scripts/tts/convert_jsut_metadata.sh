#!/bin/bash

ROOT_PATH=$1
NEOLOGD_DIR=${2:-/usr/local/lib/mecab/dic/mecab-ipadic-neologd}

pushd $ROOT_PATH

function convertjsut()
{
    cat metadata.csv | while IFS='|' read id text; 
    do
        echo -n "${id}|" >> metadata_kana.csv
        echo "${text}" \
        | mecab -Odump -d $NEOLOGD_DIR \
        | perl -lne '
            next if not m{\d+ (\S+) ([^,]+),(?:[^,]+,){4,6}([^,]+),};
            $add = ($3 eq "*" ? $1 : $3);
            if ($2 eq "助詞" or $2 eq "助動詞") {
                $f = 1; $o .= $add;
            } else {
                if ($f == 1) {
                    $r .= " $o"; $o = ""; $f = 0;
                }
                $o .= $add if $1 ne "BOS" and $1 ne "EOS";
            }
            END {
                $r = "$r $o";
                $r =~ s{^\s*(.*?)\s*$}{\1}g;
                $r =~ s{ ?、}{, }g;
                $r =~ s{ ?。}{.}g;
                print $r;
            }' >> metadata_kana.csv
    done
    ## 長音とくっついて 1 字になってしまうので最初に長音を変換する…
    cat metadata_kana.csv | perl -pe 's{ー}{-}g;' | uconv -f UTF-8 -t UTF-8 -x Latin | perl -pe 's{n'\''}{nn}g; s{~}{}g;' > metadata_alpha.csv
}

convertjsut

## 2 カラム → 3 カラム
perl -F\\\| -i.org -alne 'print "$F[0]|$F[1]|$F[1]"' metadata_alpha.csv
perl -F\\\| -i.org -alne 'print "$F[0]|$F[1]|$F[1]"' metadata_kana.csv

echo [metadata.csv]
head -n1 metadata.csv
echo
echo [metadata_alpha.csv]
head -n1 metadata_alpha.csv
echo
echo [metadata_kana.csv]
head -n1 metadata_kana.csv

popd
