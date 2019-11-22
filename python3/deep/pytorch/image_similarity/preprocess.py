import argparse
import sys
import os
import random
import shutil
import numpy as np
from tqdm import tqdm
import time
import torch
from torchvision import transforms
from PIL import Image, ImageFile
import cv2

PROCESSED_IMAGES_DIR = 'processed_images'

def remove_duplicated_images():
    print('--- remove duplicated images ---')

    n_remove_file = 0
    file_map = {}
    for file in tqdm(os.listdir(args.target_dir)):
        (name, ext) = os.path.splitext(file)
        if ext != '.png':
            continue

        splited = name.split('_')
        if len(splited) < 2:
            continue

        hash = splited[1] 
        if hash in file_map:
            path = '{}/{}'.format(args.target_dir, file)
            n_remove_file += 1
            os.remove(path)
            continue
        
        file_map[hash] = file

    print('removed: {}, remained: {}'.format(n_remove_file, len(file_map)))

    return file_map

def categorize_images():
    print('--- categorize_images images ---')
    index = 0
    groups = []
    for file in tqdm(file_map.values()):
        is_new_group = True
        path = '{}/{}'.format(args.target_dir, file)
        image = cv2.imread(path)
        for i, files in enumerate(groups):
            if args.similar_groups < i:
                break
            img = cv2.imread('{}/{}'.format(args.target_dir, files[0]))
            diff = (img - image).flatten()
            average = np.abs(diff).sum() / len(diff) / 255
            # print(file, files[0], average)
            if average < args.similar_threshold:
                files.append(file)
                is_new_group = False
                break

        if is_new_group:
            groups.append([file])

        index += 1
        if index % 10 == 0:
            index = 0
            groups = sorted(groups, key=lambda x:-len(x))

    similar_image_dir = './{}'.format(args.dest_dir)
    if os.path.exists(similar_image_dir):
        shutil.rmtree(similar_image_dir)

    os.mkdir(similar_image_dir)

    for i, images in enumerate(groups):
        if not os.path.exists('{}/{}'.format(similar_image_dir, i)):
            os.mkdir('{}/{}'.format(similar_image_dir, i))

        for file in images:
            src = os.path.realpath('{}/{}'.format(args.target_dir, file))
            dst = '{}/{}/{}'.format(similar_image_dir, i, file)
            os.symlink(src, dst)


def transform_images():
    print('--- transform images ---')
    list = [
        transforms.Grayscale(),
        transforms.Resize((144, 256)),
    ]
    forms = transforms.Compose(list)

    if not os.path.exists(PROCESSED_IMAGES_DIR):
        os.mkdir(PROCESSED_IMAGES_DIR)

    for file in tqdm(file_map.values()):
        path = '{}/{}'.format(args.target_dir, file)
        image = Image.open(path)

        forms(image).save('./{}/{}'.format(PROCESSED_IMAGES_DIR, file))

def copy_images():
    assert args.dest_dir != None, "use --dest_dir to show files to remove"

    print('--- copy images ---')

    print('copy some images in {} to {}'.format(args.target_dir, args.dest_dir))

    count = len([name for name in os.listdir(args.target_dir)])
    random_list = [random.randint(0, count) for _ in range(args.copy_count)]

    copy_dir = './{}'.format(args.dest_dir)
    os.makedirs(copy_dir, exist_ok=True)
    for i, entry  in enumerate(os.listdir(args.target_dir)):
        if i not in random_list:
            continue
        symlink = '{}/{}'.format(args.target_dir, entry)
        file = os.path.realpath(symlink)
        (name, ext) = os.path.splitext(file)
        if ext != '.png':
            continue

        shutil.copyfile(file, '{}/{}'.format(copy_dir, entry))

def remove_images():
    assert args.dest_dir != None, "use --dest_dir to show files to remove"

    print('--- remove images ---')

    remove_dir = '/tmp/remove_images'
    print('move images in {} from {} to {}'.format(args.dest_dir, args.target_dir, remove_dir))

    os.makedirs(remove_dir, exist_ok=True)
    for entry  in tqdm(os.listdir(args.dest_dir)):
        symlink = '{}/{}'.format(args.dest_dir, entry)
        file = os.path.realpath(symlink)
        (name, ext) = os.path.splitext(file)
        if ext != '.png':
            continue

        shutil.move(file, remove_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='check image similarity')
    parser.add_argument('target_dir', help='target image directory path')
    parser.add_argument('--transform', action='store_true', help='transform and save images')
    parser.add_argument('--categorize', action='store_true', help='categorize images')
    parser.add_argument('--remove', action='store_true', help='remove images')
    parser.add_argument('--copy', action='store_true', help='copy some images to keep')
    parser.add_argument('--dest_dir', default='./similar_images', help='destination directory')
    parser.add_argument('--similar_groups', type=int, default=20, help='valid group count')
    parser.add_argument('--similar_threshold', type=float, default=0.1, help='treat as simialr images when value is smaller than this')
    parser.add_argument('--copy_count', type=int, default=10, help='file count to copy')
    args = parser.parse_args()
    print(args)
 
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    file_map = remove_duplicated_images()

    if args.categorize:
        categorize_images()

    if args.remove:
        remove_images()

    if args.transform:
        transform_images()

    if args.copy:
        copy_images()
