# 名称：
#   Pascal VOC 形式のアノテーションデータをYolo V3形式に変換
#
# 前提：
#   Pascal VOCフォルダと同階層で本スクリプトを実行し、
#   その中にlabelsフォルダを作成し変換後データを格納する。
#   BASE_PATH_NAMEはPascal VOC 形式データを格納するディレクトリ名
#
# 入力ファイル：
#    BASE_PATH_NAME/ImageSets/Main/CLASS_DATASETS.txt  クラス数×データセット数
#    BASE_PATH_NAME/JPEGImages/INPUT_IMAGES.jpg        イメージファイル数
#
# 出力ファイル：
#    train.txt
#    val.txt
#    BASE_PATH_NAME/labels/IMAGE_FILENAME.txt          イメージフィアル数
#
 
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob
 
BASE_PATH_NAME = 'Output/PascalVOC-export' # PascalVOC形式データのルートディレクトリ
DATA_SETS = ['train', 'val']                    # データセットのリスト(訓練用と評価用の名称)
classes = ["class1", "class2"]            # Yoloでのclass，VoTTでのTAG
 
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
def convert_annotation(image_file):
    in_file  = open('%s/Annotations/%s.xml'%( BASE_PATH_NAME, image_file))
    out_file = open('%s/labels/%s.txt'%(BASE_PATH_NAME, image_file), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
 
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
wd = getcwd()
 
for image_set in DATA_SETS:
    # 結果を保存するディレクトリを作成
    if not os.path.exists( BASE_PATH_NAME + '/labels/'):
        os.makedirs(BASE_PATH_NAME + '/labels/')
 
    files = glob.glob(BASE_PATH_NAME + '/ImageSets/Main/*%s.txt'%(image_set))
 
    # ファイル名のリストを取得(拡張子なし)
    image_ids = open(files[0]).readlines()  # 各データセットに全ファイルが列挙されるため[0]だけで良い
    input_files = []
    for line in image_ids:
        line = line.strip().split() # スペースで分割
        line = line[0].split('.')   # .で分割
        input_files.append(line[0])
    list_file = open('%s.txt'%(image_set), 'w')
    for image_file in input_files:
        list_file.write( '%s/%s/JPEGImages/%s.jpg\n'%( wd, BASE_PATH_NAME, image_file))
        convert_annotation(image_file)
    list_file.close()
