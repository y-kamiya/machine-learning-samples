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

count_resize=$(find $SOURCE_DIR/resize -maxdepth 1 -name '*.xml' | wc -l)
count_rotate=$((count_resize / 3))
echo $count_resize
echo $count_rotate
echo =============
find $SOURCE_DIR/resize/rotate -maxdepth 1 -name '*.xml' \
| sed 's/\.xml//' | shuf | head -n $count_rotate | xargs -n1 -I@ mv @.jpg @.xml $all_dir

# mv $SOURCE_DIR/resize/rotate $all_dir
find $SOURCE_DIR/resize/ -maxdepth 1 -name '*.jpg' | xargs -I@ mv @ $all_dir
find $SOURCE_DIR/resize/ -maxdepth 1 -name '*.xml' | xargs -I@ mv @ $all_dir

pushd $all_dir

mkdir -p train
python $script_dir/annotator.py ./ --copy_images_per_class $COUNT --output_dir ./train --mv

# find . -maxdepth 1 -name '*.xml' \
# | sed 's/\.xml//' | shuf | head -n $COUNT | xargs -n1 -I@ mv @.jpg @.xml train

popd

if [ $VOC2COCO != undef ]; then
    python $VOC2COCO $all_dir/train $all_dir/train/annotations.json
fi


