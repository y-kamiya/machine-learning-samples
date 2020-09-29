import os
import cv2
import hashlib
import argparse
from tqdm import tqdm

class Preprocessor():
    MIN = 144
    MAX = 1280

    def __init__(self):
        pass

    def rotate(self, dir, file, angle):
        if not self.__is_image(file):
            return

        rotate_dir = os.path.join(dir, 'rotate')
        os.makedirs(rotate_dir, exist_ok=True)

        image = cv2.imread(os.path.join(dir, file))
        h, w, _ = image.shape

        center = (int(w/2), int(h/2))
        transform = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, transform, (w, h))

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        cv2.imwrite(self.__get_output_path(rotated, rotate_dir), rotated, encode_param)

    def clamp(self, dir, file):
        if not self.__is_image(file):
            return

        clamp_dir = os.path.join(dir, 'clamp')
        os.makedirs(clamp_dir, exist_ok=True)

        image = cv2.imread(os.path.join(dir, file))
        
        h_orig, w_orig, _ = image.shape
        ratio = h_orig / w_orig
        longer = w_orig if h_orig <= w_orig else h_orig
        shorter = w_orig if h_orig > w_orig else h_orig

        scale = 1
        if self.MAX < longer:
            scale = self.MAX / longer
        elif shorter < self.MIN:
            scale = self.MIN / shorter

        clamped = image
        if scale != 1:
            print('clamped ({}, {}) => ({}, {})'.format(w_orig, h_orig, int(w_orig * scale), int(h_orig * scale)))
            clamped = cv2.resize(image, dsize=None, fx=scale, fy=scale)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        cv2.imwrite(self.__get_output_path(clamped, clamp_dir), clamped, encode_param)

    def __is_image(self, path):
        _, ext = os.path.splitext(path)
        return ext in ['.jpg', '.jpeg', '.png']

    def __get_output_path(self, image, output_dir):
        md5 = hashlib.md5(image).hexdigest()
        return '{}/{}.jpg'.format(output_dir, md5, 'jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess images')
    parser.add_argument('target', default='./images', help='path to target file or directory that has target images')
    parser.add_argument('--clamp', action='store_true', help='clamp images')
    parser.add_argument('--rotate', type=int, default=0, help='create rotate images')
    args = parser.parse_args()

    preprocessor = Preprocessor()

    if args.rotate != 0:
        for file in tqdm(os.listdir(args.target)):
            preprocessor.rotate(args.target, file, args.rotate)

    if args.clamp:
        for file in tqdm(os.listdir(args.target)):
            preprocessor.clamp(args.target, file)
