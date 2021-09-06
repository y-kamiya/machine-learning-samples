#!/bin/sh

mkdir data
pushd data

# wikipedia2vec
curl -L -O "http://wikipedia2vec.s3.amazonaws.com/models/ja/2018-04-20/jawiki_20180420_300d.pkl.bz2"
tar jxf jawiki_20180420_300d.pkl.bz2

# wordnet
curl -L -O "http://compling.hss.ntu.edu.sg/wnja/data/1.1/wnjpn.db.gz"
gunzip wnjpn.db.gz

popd
