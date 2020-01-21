import os
import sys
import glob
import shutil
import argparse
import csv
import cv2
import hashlib
import random
import numpy as np
from tqdm import tqdm
import imghdr

# utils
def is_ext(path, ext):
    _, str = os.path.splitext(os.path.basename(path))
    return str == ext

def parse_filename(path):
    file = os.path.basename(path)
    name, _ = os.path.splitext(file)
    parts = name.rsplit('@', 1)
    labels = parts[0].split('@')
    return {'hash':parts[1], 'labels':labels, 'file':file}



def extract_label():
    if is_ext(args.target_dir, '.csv'):
        with open(args.target_dir) as f:
            reader = csv.reader(f)
            for entry in reader:
                print(entry[5].split('@')[0])
        sys.exit()

    for root, dirs, files in os.walk(args.target_dir):
        for file in files:
            if not is_ext(file, '.jpg'):
                continue
            parsed = parse_filename(file)
            print(parsed['labels'][0])

def classify_with_label():
    assert args.target_label != None, "use --target_label to glob files"
    assert args.dest_dir != None, "use --dest_dir to show directory to put images"

    target = '{}/{}*'.format(args.target_dir, args.target_label)
    for path in glob.glob(target):
        parsed = parse_filename(file)
        label = parsed['labels'][0]
        label_dir = '{}/{}'.format(args.dest_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        count = len([f for f in os.listdir(label_dir)]) 
        if count < args.count:
            shutil.copyfile(path, '{}/{}'.format(label_dir, parsed['file']))

def create_grayscale_images():
    assert args.dest_dir != None, "use --dest_dir to save processed images"
    print('--- grayscale images ---')

    os.makedirs(args.dest_dir, exist_ok=True)

    for file in tqdm(os.listdir(args.target_dir)):
        if is_ext(file, '.jpg') or is_ext(file, '.png'):
            continue

        path = '{}/{}'.format(args.target_dir, file)
        image = cv2.imread(path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('{}/{}'.format(args.dest_dir, file), image_gray)

def create_csv():
    output_file = '{}/labels.csv'.format(args.target_dir)
    data = []
    for root, _, files in os.walk(args.target_dir):
        target_dir_abs = os.path.realpath(root)
        for file in files:
            if not is_ext(file, '.jpg'):
                continue
            parsed = parse_filename(file)
            labels = parsed['labels'][0].replace('@', ',')
            path = '{}/{}'.format(target_dir_abs, file)
            data.append('{},{}'.format(path, labels))
    
    with open(output_file, mode='w') as f:
        f.write('\n'.join(data))

def extract_node_image():
    node_data_file = os.path.join(args.target_dir, args.extract_node_image)
    assert os.path.exists(node_data_file), "wrong file is passed to --extract_node_image"

    os.makedirs('{}/node'.format(args.target_dir), exist_ok=True)

    images = {}
    for file in os.listdir(args.target_dir):
        splited = os.path.splitext(file)
        if splited[1] != '.png':
            continue
        hash = splited[0].split('_')[1]
        images[hash] = file

    dest_dir = os.path.join(args.target_dir, 'node')
    os.makedirs(os.path.join(dest_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'crop'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'wide'), exist_ok=True)

    with open(node_data_file) as f:
        reader = csv.reader(f)
        current_hash = ''
        for entry in tqdm(list(reader)):
            hash = entry[0]
            if hash not in images:
                continue

            if hash != current_hash:
                image = None
                current_hash = hash
                path = '{}/{}'.format(args.target_dir, images[hash])
                if os.path.exists(path):
                    image = cv2.imread(path)

            if image is None:
                continue

            x = max(int(entry[1]), 0)
            y = max(int(entry[2]), 0)
            w = int(entry[3])
            h = int(entry[4])
            label = entry[5].split('@')[0]

            height, width, _ = image.shape
            cv_y = height - y
            save_image(image[cv_y-h:cv_y, x:x+w], label, 'raw', dest_dir)
            if h < 720 and w < 1280:
                # 20%外側も含める
                _h = int(h * 0.2)
                _w = int(w * 0.2)
                y_from = max(0, cv_y - h - _h)
                y_to = min(cv_y + _h, 720)
                x_from = max(0, x - _w)
                x_to = min(x + w + _w, 1280)
                save_image(image[y_from:y_to, x_from:x_to], label, 'wide', dest_dir)
            if 800 < w:
                # 横に半分
                save_image(image[cv_y-h:cv_y, x:x+w//2], label, 'crop', dest_dir)
                save_image(image[cv_y-h:cv_y, x+w//2:x+w], label, 'crop', dest_dir)
            if 600 < h:
                # 縦に半分
                save_image(image[cv_y-h:cv_y-h//2, x:x+w], label, 'crop', dest_dir)
                save_image(image[cv_y-h//2:cv_y-h, x:x+w], label, 'crop', dest_dir)
            if 800 < w and 600 < h:
                # 中央
                save_image(image[cv_y-h*3//4:cv_y-h//4, x+w//4:x+w*3//4], label, 'crop', dest_dir)
            if 1280 <= w and 720 <= w:
                # 全画面の場合は４分割も
                save_image(image[cv_y-h:cv_y-h//2, x:x+w//2], label, 'crop', dest_dir)
                save_image(image[cv_y-h:cv_y-h//2, x+w//2:x+w], label, 'crop', dest_dir)
                save_image(image[cv_y-h//2:cv_y, x:x+w//2], label, 'crop', dest_dir)
                save_image(image[cv_y-h//2:cv_y, x+w//2:x+w], label, 'crop', dest_dir)

def augmentation():
    assert args.target_label != None, "use --target_label to show class to apply augmentation"
    assert args.augmentation in ['gamma', 'resize'], "use --target_label to show class to apply augmentation"

    dest_dir = os.path.join(args.target_dir, '..', 'aug_' + args.augmentation)

    labels = [args.target_label]
    if os.path.exists(args.target_label):
        with open(args.target_label, 'r') as f:
            content = f.read()
            labels = content.split('\n')
            labels.pop(-1) # remove empty line

    for label in tqdm(labels):
        src_dirs = []
        for root, dirs, _ in os.walk(args.target_dir):
            if label in dirs:
                src_dirs.append(os.path.join(root, label))

        for src_dir in src_dirs:
            if args.augmentation == 'gamma':
                augmentation_gamma(src_dir, dest_dir, label)
            elif args.augmentation == 'resize':
                augmentation_resize(src_dir, dest_dir, label)

def augmentation_gamma(src_dir, dest_dir, label):
    assert args.target_label != None, "use --target_label to show class to apply augmentation"

    gammas = [round(i * 0.1, 1) for i in range(7, 14, 2)]

    for file in os.listdir(src_dir):
        image = cv2.imread(os.path.join(src_dir, file))
        if image is None:
            continue
        for gamma in gammas:
            gamma_cvt = np.zeros((256,1),dtype = 'uint8')
            for i in range(256):
                gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)
            im = cv2.LUT(image, gamma_cvt)

            save_image(im, label, '', dest_dir)

def augmentation_resize(src_dir, dest_dir, label):
    assert args.target_label != None, "use --target_label to show class to apply augmentation"

    ratios = [0.5, 0.75, 1.25, 1.5]

    for file in os.listdir(src_dir):
        image = cv2.imread(os.path.join(src_dir, file))
        if image is None:
            continue
        h, w, _ = image.shape

        for r in ratios:
            _w, _h = int(w * r), int(h * r)
            if _w < 30 or _h < 30:
                continue
            im = cv2.resize(image, (_w, _h), interpolation=cv2.INTER_CUBIC)

            save_image(im, label, '', dest_dir)

def save_image(image, label, type, dest_dir):
    dest_dir = os.path.join(dest_dir, type, label)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    hash = hashlib.md5(image.tobytes()).hexdigest()
    path = '{}/{}@{}.jpg'.format(dest_dir, label, hash)
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 85])

def convert_to_imagenet_structure():
    assert args.dest_dir != None, "use --dest_dir to show directory to put images"

    train_dir = os.path.join(args.dest_dir, 'train')
    if os.path.isdir(train_dir):
        print('dest_dir has already exist')
        return

    os.makedirs(train_dir, exist_ok=True)

    for dir in tqdm(os.listdir(args.target_dir)):
        path = os.path.join(args.target_dir, dir)
        if not os.path.isdir(path):
            continue
        shutil.move(path, train_dir)
        # parsed = parse_filename(file)
        # label = parsed['labels'][0]
        # label_dir = '{}/{}'.format(train_dir, label)
        # os.makedirs(label_dir, exist_ok=True)
        # path = '{}/{}'.format(args.target_dir, file)
        # shutil.copyfile(path, '{}/{}'.format(label_dir, file))

    if args.validation_ratio == None:
        return

    val_dir = os.path.join(args.dest_dir, 'validation')
    os.makedirs(val_dir, exist_ok=True)

    for label in tqdm(os.listdir(train_dir)):
        label_dir = os.path.join(train_dir, label)
        if not os.path.isdir(label_dir):
            continue
        count = int(len([f for f in os.listdir(label_dir)]) * args.validation_ratio)
        for i, file in enumerate(os.listdir(label_dir)):
            if count <= i:
                break
            src = os.path.join(label_dir, file)
            tgt = os.path.join(val_dir, file)
            shutil.move(src, tgt)

def create_node_uniq():
    assert is_ext(args.target_dir, '.csv'), 'pass csv file that has node positions'
    assert os.path.exists(args.target_dir), 'target file is not found'

    with open(args.target_dir) as f:
        reader = csv.reader(f)

        map_uniq = {}
        map_index = {}
        for index, entry in enumerate(reader):
            str = ','.join(entry)
            if str in map_uniq:
                continue
            map_uniq[str] = index

            label = entry[5].split('@')[0]
            if label not in map_index:
                map_index[label] = []
            map_index[label].append(index)

    map_uniq_reversed = {v:k for k,v in map_uniq.items()}
    dir = os.path.dirname(args.target_dir)
    output_file = os.path.join(dir, 'node_uniq.csv')
    with open(output_file, 'w') as f:
        for indexes in map_index.values():
            samples = random.sample(indexes, min(len(indexes), args.count))
            for index in samples:
                f.write(map_uniq_reversed[index])
                f.write('\n')

def check_jpg():
    for root, _, files in os.walk(args.target_dir):
        for file in files:
            path = os.path.join(root, file)
            type = imghdr.what(path)
            if type == None:
                print(path)

def resize_images():
    assert args.dest_dir != None, "need --dest_dir"

    ratio = 0.5

    for file in tqdm(os.listdir(args.target_dir)):
        image = cv2.imread(os.path.join(args.target_dir, file))
        if image is None:
            continue
        h, w, _ = image.shape
        _w, _h = int(w * ratio), int(h * ratio)
        im = cv2.resize(image, (_w, _h), interpolation=cv2.INTER_CUBIC)

        parsed = parse_filename(file)
        save_image(im, parsed['labels'][0], '', args.dest_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='check image similarity')
    parser.add_argument('target_dir', help='target image directory path')
    parser.add_argument('--extract_label', action='store_true', help='extract label')
    parser.add_argument('--extract_node_image', default=None, help='extract image from screenshot according to csv')
    parser.add_argument('--classify_with_label', action='store_true', help='classify images with label in file name')
    parser.add_argument('--grayscale', action='store_true', help='get grayscale images')
    parser.add_argument('--csv', action='store_true', help='create csv file including map between images and labels')
    parser.add_argument('--count', type=int, default=10, help='file count to process')
    parser.add_argument('--target_label', default=None, help='label to process')
    parser.add_argument('--dest_dir', default=None, help='destination directory')
    parser.add_argument('--convert_imagenet', action='store_true', help='convert directory structure like imagenet')
    parser.add_argument('--validation_ratio', type=float, default=None, help='ratio of validation images')
    parser.add_argument('--create_node_uniq', action='store_true', help='create node_uniq.csv')
    parser.add_argument('--augmentation', default=None, help='caugmentation type: gamma, resize')
    parser.add_argument('--check_jpg', action='store_true', help='check if jpg images exist')
    parser.add_argument('--resize', action='store_true', help='resize images')
    args = parser.parse_args()
    # print(args)

    if args.extract_label:
        extract_label()
        sys.exit()

    if args.extract_node_image != None:
        extract_node_image()
        sys.exit()

    if args.classify_with_label:
        classify_with_label()
        sys.exit()

    if args.grayscale:
        create_grayscale_images()
        sys.exit()

    if args.csv:
        create_csv()
        sys.exit()

    if args.convert_imagenet:
        convert_to_imagenet_structure()
        sys.exit()

    if args.create_node_uniq:
        create_node_uniq()
        sys.exit()

    if args.augmentation != None:
        augmentation()
        sys.exit()

    if args.check_jpg:
        check_jpg()
        sys.exit()

    if args.resize:
        resize_images()
        sys.exit()

