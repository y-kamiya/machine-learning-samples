#!/bin/bash

classlist=$1
from=$2
to=$3
count=$4

for classname in $(cat $classlist);
do
    find $from -name "$classname@*" | shuf | head -n $count | xargs -I@ cp @ $to
done
