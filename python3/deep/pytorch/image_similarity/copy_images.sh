#!/bin/bash -e

classlist=$1
from=$2
to=$3
count=$4

cat $classlist | gxargs -I{} bash -c "mkdir -p $to/{}"
cat $classlist | gxargs -n1 -I{} -P8 bash -c "find $from -maxdepth 3 -name \"{}@*\" | shuf | head -n $count | gxargs -I@ -P8 cp @ $to/{}/"
#head -n $count | gxargs -I@ -P8 echo @ $to/{}"
