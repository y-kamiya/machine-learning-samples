#!/bin/bash

SOURCE_DIR=$1
COUNT=$2
VOC2COCO=${3:-undef}

script_dir=$(cd `dirname $0`; pwd)
all_dir=$SOURCE_DIR/../all

python $script_dir/augmentator.py --resize $SOURCE_DIR
python $script_dir/augmentator.py --rotate $SOURCE_DIR/resize

rm -rf $all_dir
mkdir -p $all_dir

count_resize=$(ls $SOURCE_DIR/resize | wc -l)
count_rotate=$((count_resize / 3))
echo $count_resize
echo $count_rotate
echo =============
find $SOURCE_DIR/resize/rotate -maxdepth 1 -name '*.jpg' \
| sed 's/\.jpg//' | shuf | head -n $count_rotate | xargs -n1 -I@ mv @.jpg @.xml $all_dir

# mv $SOURCE_DIR/resize/rotate $all_dir
mv $SOURCE_DIR/resize/*.jpg $all_dir
mv $SOURCE_DIR/resize/*.xml $all_dir

pushd $all_dir

mkdir -p train
find . -maxdepth 1 -name '*.jpg' \
| sed 's/\.jpg//' | shuf | head -n $COUNT | xargs -n1 -I@ mv @.jpg @.xml train

popd

if [ $VOC2COCO != undef ]; then
    python $VOC2COCO $all_dir/train $all_dir/train/annotations.json
fi


