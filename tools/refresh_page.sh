#!/bin/bash

URL=$1
SECONDS=${2:-1800}

if [ -z $URL ]; then
    echo "pass url as first argument"
    exit
fi

for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open -n -a 'Google Chrome' $URL
  # open -n -a 'Google Chrome' --args --incognito $URL
  sleep $SECONDS
done
