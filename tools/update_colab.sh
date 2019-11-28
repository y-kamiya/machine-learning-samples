#!/bin/bash

URL=$1

if [ -z $URL ]; then
    echo "pass url as first argument"
    exit
fi

for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open $URL
  sleep 3600
done
