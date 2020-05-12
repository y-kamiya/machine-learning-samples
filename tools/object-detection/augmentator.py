import os
import cv2
import hashlib
import argparse
import albumentations as alb
from tqdm import tqdm
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET

class Augmentator():
    MIN_BBOX = 16

    def __init__(self, config):
        self.config = config

    def execute(self, path):
        if not self.__is_xml(path):
            return

        name, _ = os.path.splitext(path)
        image = cv2.imread('{}.jpg'.format(name))

        annotations = self.__build_annotations(image, path)

        if self.config.rotate:
            rotate_dir = os.path.join(os.path.dirname(path), 'rotate')
            for _ in range(2):
                self.__save_result([alb.Rotate(limit=135, always_apply=True)], annotations, rotate_dir)

        if self.config.flip:
            flip_dir = os.path.join(os.path.dirname(path), 'flip')
            self.__save_result([alb.HorizontalFlip(p=1)], annotations, flip_dir)

        if self.config.resize:
            resize_dir = os.path.join(os.path.dirname(path), 'resize')
            for w, h in self.__get_resize_patterns(image):
                self.__save_result([alb.Resize(p=1, width=w, height=h)], annotations, resize_dir)

    def __is_xml(self, path):
        _, ext = os.path.splitext(path)
        return ext in ['.xml']

    def __get_resize_patterns(self, image):
        height, width, _ = image.shape
        min = 144
        max = 1280
        resized_ratio = [0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.5, 2, 4, 8]

        patterns = []
        for ratio in resized_ratio:
            w = int(width * ratio)
            h = int(height * ratio)
            if min <= w and w <= max and min <= h and h <= max:
                patterns.append((w, h))

        return patterns

    def __build_annotations(self, image, xml_path):
        with open(xml_path, 'r') as f:
            xml = ET.fromstring(f.read())

        labels = []
        bboxes = []
        for obj in xml.findall('object'):
            labels.append(obj.find('name').text)
            bboxes.append([int(e.text) for e in obj.find('bndbox')])

        return {'image': image, 'bboxes': bboxes, 'category_id': labels}

    def __save_result(self, aug, annotations, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        compose = alb.Compose(aug, bbox_params=alb.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['category_id']))

        result = compose(**annotations)

        output_path = self.__get_output_path(result['image'], output_dir)
        filename, _ = os.path.splitext(output_path)
        height, width, channel = result['image'].shape

        cv2.imwrite(output_path, result['image'], [cv2.IMWRITE_JPEG_QUALITY, 100])

        writer = Writer(output_path, width, height, channel)
        for i in range(len(result['bboxes'])):
            e = [int(value) for value in result['bboxes'][i]]
            if e[2] - e[0] < self.MIN_BBOX or e[3] - e[1] < self.MIN_BBOX:
                continue
            writer.addObject(result['category_id'][i], e[0], e[1], e[2], e[3])

            # image_tmp = result['image'].copy()
            # cv2.rectangle(image_tmp, (e[0], e[1]), (e[2], e[3]), (0, 0, 255), 2)
            # cv2.imshow("annotation", image_tmp)
            # cv2.waitKey(0)

        writer.save('{}.xml'.format(filename))

    def __get_output_path(self, image, output_dir):
        md5 = hashlib.md5(image).hexdigest()
        return '{}/{}.jpg'.format(output_dir, md5, 'jpg')

    def __resize(self, image, xml):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='annotate images')
    parser.add_argument('target', default='./images', help='path to target file or directory that has target images')
    parser.add_argument('--flip', action='store_true', help='apply flip')
    parser.add_argument('--resize', action='store_true', help='apply resize')
    parser.add_argument('--rotate', action='store_true', help='apply rotate')
    args = parser.parse_args()

    augmentator = Augmentator(args)

    for file in tqdm(os.listdir(args.target)):
        augmentator.execute(os.path.join(args.target, file))
