import argparse
import sys
import os
import pickle
from tqdm import tqdm
import time
import torch
from torchvision import transforms
from PIL import Image, ImageFile

PROCESSED_IMAGES_DIR = 'processed_images'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='check image similarity')
    parser.add_argument('target_dir', help='target image directory path')
    args = parser.parse_args()
    print(args)
 
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    print('--- remove duplicated files ---')
    n_remove_file = 0
    file_map = {}
    remove_files = []
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

    print('--- transform files ---')
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
        


