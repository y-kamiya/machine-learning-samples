#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""feature detection."""

import cv2
import os
import sys
import argparse
import numpy as np


def check_features(args):
    # IMG_SIZE = (256, 144)

    target_img_path = args.target
    target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
    # target_img = cv2.resize(target_img, IMG_SIZE)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # detector = cv2.ORB_create()
    detector = cv2.AKAZE_create()
    (target_kp, target_des) = detector.detectAndCompute(target_img, None)

    files = os.listdir(args.target_dir)
    for file in files:
        _, ext = os.path.splitext(file)
        if ext != '.png':
            continue

        comparing_img_path = '{}/{}'.format(args.target_dir, file)
        try:
            comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
            # comparing_img = cv2.resize(comparing_img, IMG_SIZE)
            (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)
            matches = bf.match(target_des, comparing_des)
            dist = [m.distance for m in matches]
            ret = sum(dist) / len(dist)
        except cv2.error:
            ret = 100000

        print(file, ret)

def check_pixels(args):
    target_img_path = args.target
    target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
    print('base image: {}'.format(target_img_path))

    files = os.listdir(args.target_dir)
    for file in files:
        _, ext = os.path.splitext(file)
        if ext != '.png':
            continue

        comparing_img_path = '{}/{}'.format(args.target_dir, file)
        comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)

        diff = (comparing_img - target_img).flatten()
        average = np.abs(diff).sum() / len(diff) / 255
        print('{} {:.3f}'.format(file, average))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='check image similarity')
    parser.add_argument('type', default='feature', help='execution type')
    parser.add_argument('target', metavar='file', help='target file path')
    parser.add_argument('--target-dir', default='./', metavar='dir', help='directory path that has comparing files')
    args = parser.parse_args()
    print(args)

    if args.type == 'pix':
        check_pixels(args)
        sys.exit()

    check_features(args)

