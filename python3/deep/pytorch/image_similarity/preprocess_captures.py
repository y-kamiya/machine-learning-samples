import os
import sys
import glob
import shutil
import argparse
import csv
import cv2
import hashlib
from tqdm import tqdm

def extract_label():
    for file in os.listdir(args.target_dir):
        labels = file.rsplit('_', 1)[0]
        label = labels.split('@')[0]
        print(label)

def classify_with_label():
    assert args.target_label != None, "use --target_label to glob files"
    assert args.dest_dir != None, "use --dest_dir to show directory to put images"

    target = '{}/{}*'.format(args.target_dir, args.target_label)
    for path in glob.glob(target):
        file = os.path.basename(path)
        label = file.rsplit('_', 1)[0]
        label_dir = '{}/{}'.format(args.dest_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        count = len([f for f in os.listdir(label_dir)]) 
        if count < args.count:
            shutil.copyfile(path, '{}/{}'.format(label_dir, file))

def create_grayscale_images():
    assert args.dest_dir != None, "use --dest_dir to save processed images"
    print('--- grayscale images ---')

    os.makedirs(args.dest_dir, exist_ok=True)

    for file in tqdm(os.listdir(args.target_dir)):
        (name, ext) = os.path.splitext(file)
        if ext not in ['.png', '.jpg']:
            continue

        path = '{}/{}'.format(args.target_dir, file)
        image = cv2.imread(path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('{}/{}'.format(args.dest_dir, file), image_gray)

def create_csv():
    output_file = '{}/labels.csv'.format(args.target_dir)
    target_dir_abs = os.path.realpath(args.target_dir)
    data = []
    for file in os.listdir(args.target_dir):
        labels = file.rsplit('_', 1)[0]
        labels = labels.replace('@', ',')
        path = '{}/{}'.format(target_dir_abs, file)
        data.append('{},{}'.format(path, labels))
    
    with open(output_file, mode='w') as f:
        f.write('\n'.join(data))

def extract_node_image():
    os.makedirs('{}/node'.format(args.target_dir), exist_ok=True)
    os.makedirs('{}/processed'.format(args.target_dir), exist_ok=True)

    images = {}
    for file in os.listdir(args.target_dir):
        splited = os.path.splitext(file)
        if splited[1] != '.png':
            continue
        hash = splited[0].split('_')[1]
        images[hash] = file

    node_data_file = '{}/node_uniq.csv'.format(args.target_dir)
    with open(node_data_file) as f:
        reader = csv.reader(f)
        current_hash = ''
        for entry in tqdm(reader):
            hash = entry[0]
            if hash not in images:
                continue

            if hash != current_hash:
                image = None
                current_hash = hash
                path = '{}/{}'.format(args.target_dir, images[hash])
                if os.path.exists(path):
                    image = cv2.imread(path)
                    # dir = '{}/processed/'.format(args.target_dir)
                    # shutil.move(path, dir)

            if image is None:
                continue

            x = max(int(entry[1]), 0)
            y = max(int(entry[2]), 0)
            w = int(entry[3])
            h = int(entry[4])
            label = entry[5].split('@')[0]

            os.makedirs('{}/node/{}'.format(args.target_dir, 'raw'), exist_ok=True)
            os.makedirs('{}/node/{}'.format(args.target_dir, 'crop'), exist_ok=True)
            os.makedirs('{}/node/{}'.format(args.target_dir, 'wide'), exist_ok=True)

            height, width, _ = image.shape
            cv_y = height - y
            save_image(image[cv_y-h:cv_y, x:x+w], label, 'raw')
            if h < 576 and w < 1024:
                # 20%外側も含める
                _h = int(h * 0.2)
                _w = int(w * 0.2)
                y_from = max(0, cv_y - h - _h)
                y_to = min(cv_y + _h, 720)
                x_from = max(0, x - _w)
                x_to = min(x + w + _w, 1280)
                save_image(image[y_from:y_to, x_from:x_to], label, 'wide')
            if 800 < w:
                # 横に半分
                save_image(image[cv_y-h:cv_y, x:x+w//2], label, 'crop')
                save_image(image[cv_y-h:cv_y, x+w//2:x+w], label, 'crop')
            if 600 < h:
                # 縦に半分
                save_image(image[cv_y-h:cv_y-h//2, x:x+w], label, 'crop')
                save_image(image[cv_y-h//2:cv_y-h, x:x+w], label, 'crop')
            if 800 < w and 600 < h:
                # 中央
                save_image(image[cv_y-h*3//4:cv_y-h//4, x+w//4:x+w*3//4], label, 'crop')
            if 1280 <= w and 720 <= w:
                # 全画面の場合は４分割も
                save_image(image[cv_y-h:cv_y-h//2, x:x+w//2], label, 'crop')
                save_image(image[cv_y-h:cv_y-h//2, x+w//2:x+w], label, 'crop')
                save_image(image[cv_y-h//2:cv_y, x:x+w//2], label, 'crop')
                save_image(image[cv_y-h//2:cv_y, x+w//2:x+w], label, 'crop')


def save_image(image, label, dir):
    hash = hashlib.md5(image.tobytes()).hexdigest()
    path = '{}/node/{}/{}_{}.jpg'.format(args.target_dir, dir, label, hash)
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='check image similarity')
    parser.add_argument('target_dir', help='target image directory path')
    parser.add_argument('--extract_label', action='store_true', help='extract label')
    parser.add_argument('--extract_node_image', action='store_true', help='extract image from screenshot')
    parser.add_argument('--classify_with_label', action='store_true', help='classify images with label in file name')
    parser.add_argument('--grayscale', action='store_true', help='get grayscale images')
    parser.add_argument('--csv', action='store_true', help='create csv file including map between images and labels')
    parser.add_argument('--count', type=int, default=10, help='file count to process')
    parser.add_argument('--target_label', default=None, help='label to process')
    parser.add_argument('--dest_dir', default=None, help='destination directory')
    args = parser.parse_args()
    print(args)

    if args.extract_label:
        extract_label()
        sys.exit()

    if args.extract_node_image:
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

