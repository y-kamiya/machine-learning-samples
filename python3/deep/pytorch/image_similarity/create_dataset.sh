#!/bin/bash -e

# extract_node_image
# 極めて少ないもので簡単に取れるものは手動でスクショ
# (raw, wide, cropからの混合割合に応じてサンプリング, cropとそれ以外で1:1でいい気がする)

# augmentation用のimage抽出(250枚)
# 全体にaug_resize 
# 枚数の少ないものだけaug_gamma & aug_resize
# datasetとしてコピー
#  raw, wideから1000
#  cropから500
#  captures_singleから500
#  aug_resizeから1000
#  枚数が1000以下のもののみaug_gamma&resizeしたものからコピー(700枚以下くらいのもの)
# 壊れたimageを削除

# tf recordを生成
# upload

INPUT_DIR=$1

SCRIPT_DIR=$(cd `dirname $0` && pwd)
WORKING_DIR=$SCRIPT_DIR/work
DATASET_DIR=$WORKING_DIR/dataset

function create_classnumlist()
{
    input=$1
    python preprocess_captures.py --extract_label $input | sort | uniq -c | sort -nr | grep -v Namespace
}

function remove_invalid_images()
{
    input=$1
    find $input -name '*.png' -size -100c | xargs rm -f
    find $input -name '*.jpg' -size -100c | xargs rm -f
    find $input -name '*.jpg' | xargs identify -format "%i %w %h\n" | awk '$2 < 16 || $3 < 16 {print $0}' | xargs rm -f
}

if [ -e $WORKING_DIR ]; then
    echo "$WORKING_DIR has already existed"
    exit
fi
mkdir -p $WORKING_DIR
mkdir -p $DATASET_DIR

echo 'remove invalid png and jpg'
remove_invalid_images $INPUT_DIR

echo 'create class list'
create_classnumlist $INPUT_DIR > $WORKING_DIR/classnumlist
cat $WORKING_DIR/classnumlist | awk '{print $2}' > $WORKING_DIR/classlist
cat $WORKING_DIR/classnumlist | awk '$1 < 60 {print $2}' > $WORKING_DIR/classlist_low

echo 'extract images for augmentation'
$SCRIPT_DIR/copy_images.sh $WORKING_DIR/classlist $INPUT_DIR ${DATASET_DIR}_aug 250

echo 'execute augmentation'
python preprocess_captures.py --augmentation resize --target_label work/classlist work/dataset_aug

python preprocess_captures.py --augmentation gamma --target_label work/classlist_low work/dataset_aug
python preprocess_captures.py --augmentation gamma --target_label work/classlist_low work/aug_resize

echo 'create dataset'
$SCRIPT_DIR/copy_images.sh $WORKING_DIR/classlist $INPUT_DIR/raw $DATASET_DIR 500
$SCRIPT_DIR/copy_images.sh $WORKING_DIR/classlist $INPUT_DIR/wide $DATASET_DIR 500
$SCRIPT_DIR/copy_images.sh $WORKING_DIR/classlist $INPUT_DIR/crop $DATASET_DIR 500
$SCRIPT_DIR/copy_images.sh $WORKING_DIR/classlist $WORKING_DIR/aug_resize $DATASET_DIR 1000
$SCRIPT_DIR/copy_images.sh $WORKING_DIR/classlist $WORKING_DIR/aug_gamma $DATASET_DIR 1000

echo 'remove broken images'
python preprocess_captures.py --check_jpg $DATASET_DIR | grep -v Namespace | xargs rm -f
remove_invalid_images $DATASET_DIR

echo 'separate validation data for efficientnet training'
rm -rf $WORKING_DIR/dataset_imagenet
python preprocess_captures.py --convert_imagenet --dest_dir $DATASET_DIR --validation_ratio 0.1 $DATASET_DIR
python preprocess_captures.py --extract_label $DATASET_DIR/validation | sort > $DATASET_DIR/synset_labels.txt

echo 'create csv file for automl'
python preprocess_captures.py --csv $DATASET_DIR

