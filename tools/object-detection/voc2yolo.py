import xml.etree.ElementTree as ET
import pickle
import os
import glob
import argparse
from os import listdir, getcwd
from os.path import join
from tqdm import tqdm
 
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
 
class_list = []

def convert_annotation(input_path, output_path):
    tree = ET.parse(input_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
 
    with open(output_path, 'w') as f:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in class_list:
                class_list.append(cls)
            cls_id = class_list.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('input_dir', help='')
    parser.add_argument('--output_dir', default='labels', help='')
    args = parser.parse_args()

    wd = getcwd()
     
    os.makedirs(args.output_dir, exist_ok=True)
 
    input_paths = glob.glob(os.path.join(args.input_dir, '*.xml'))
 
    for input_path in tqdm(input_paths):
        name, _ = os.path.splitext(os.path.basename(input_path))
        output_path = '{}/{}.txt'.format(args.output_dir, name)
        convert_annotation(input_path, output_path)

    print('nc: {}'.format(len(class_list)))
    print(class_list)
 
