import os
import cv2
import hashlib
import argparse
from tqdm import tqdm

class Preprocessor():
    MIN = 80
    MAX = 960

    def __init__(self):
        pass

    def clamp(self, dir, file):
        if not self.__is_image(file):
            return

        clamp_dir = os.path.join(dir, 'clamp')
        os.makedirs(clamp_dir, exist_ok=True)

        image = cv2.imread(os.path.join(dir, file))
        
        h_orig, w_orig, _ = image.shape
        ratio = h_orig / w_orig
        need_clamp = True
        if ratio < 1.0 and h_orig < self.MIN:
            # 短辺がhでmin以下の場合
            h = self.MIN
            w = self.MIN / ratio
        elif ratio > 1.0 and w_orig < self.MIN:
            # 短辺がwでmin以下の場合
            h = self.MIN * ratio
            w = self.MIN
        elif ratio < 1.0 and w_orig > self.MAX:
            # 長辺がwでmax以上の場合
            h = self.MAX * ratio
            w = self.MAX
        elif ratio > 1.0 and h_orig > self.MAX:
            # 長辺がhでmax以上の場合
            h = self.MAM
            w = self.MAM / ratio
        else:
            need_clamp = False

        clamped = image
        if need_clamp:
            print('clamped ({}, {}) => ({}, {})'.format(w_orig, h_orig, w, h))
            clamped = cv2.resize(image, (int(w), int(h)))

        cv2.imwrite(self.__get_output_path(clamped, clamp_dir), clamped)

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
    args = parser.parse_args()

    preprocessor = Preprocessor()

    if args.clamp:
        for file in tqdm(os.listdir(args.target)):
            preprocessor.clamp(args.target, file)
