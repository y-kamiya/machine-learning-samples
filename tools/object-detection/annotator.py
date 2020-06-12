import cv2
import sys
import os.path
import argparse
import hashlib
import random
import shutil
from tqdm import tqdm
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET

script_dir = os.path.dirname(os.path.realpath(__file__))

class Annotator():

    CASCADE_FILE = os.path.join(script_dir, 'lbpcascade_animeface.xml')
    KEY_LABEL_MAP = {
        '0': 'asuna',
        '1': 'administrator',
        '2': 'kirito',
        '3': 'alice',
        '4': 'yui',
    }

    def __init__(self, config):
        self.config = config

        target = config.target
        if os.path.isfile(target):
            self.target = target
            self.target_dir = os.path.dirname(target)
            self.output_dir = os.path.splitext(target)[0]
        else:
            self.target = None
            self.target_dir = target
            self.output_dir = os.path.join(self.target_dir, 'annotations')

        if config.output_dir is not None:
            self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def annotate(self, path):
        if not os.path.isfile(self.CASCADE_FILE):
            raise RuntimeError("%s: not found" % self.CASCADE_FILE)

        if self.__is_image(path):
            self.annotate_image(path)
            return

    def detect_faces(self, image):
        cascade = cv2.CascadeClassifier(self.CASCADE_FILE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        return cascade.detectMultiScale(gray,
                                        # detector options
                                        scaleFactor = 1.1,
                                        minNeighbors = 1,
                                        minSize = (24, 24))
    def annotate_image(self, path, image=None):
        if image is None:
            image = cv2.imread(path, cv2.IMREAD_COLOR)

        faces = self.detect_faces(image)

        output_path = self.__get_output_path(image, path=path)
        shape = image.shape
        writer = Writer(output_path, shape[1], shape[0], shape[2])

        saved_count = 0
        for (x, y, w, h) in faces:
            xmax = x + w
            ymax = y + h
            image_tmp = image.copy()
            cv2.rectangle(image_tmp, (x, y), (xmax, ymax), (0, 0, 255), 2)
            cv2.imshow("annotation", image_tmp)

            label = self.get_label()
            if label is not None:
                writer.addObject(label, x, y, xmax, ymax)
                saved_count = saved_count + 1

        if 0 < saved_count:
            if path != output_path:
                cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            name = os.path.splitext(output_path)
            writer.save('{}.xml'.format(name[0]))

    def get_label(self):
        if self.config.all_faces:
            return 'face'

        key = chr(cv2.waitKey(0))
        if key not in self.KEY_LABEL_MAP:
            return None

        return self.KEY_LABEL_MAP[key]

    def __is_image(self, path):
        _, ext = os.path.splitext(path)
        return ext in ['.jpg', '.jpeg', '.png']

    def __is_movie(self, path):
        _, ext = os.path.splitext(path)
        return ext in ['.mp4', '.mov']

    def __is_xml(self, path):
        _, ext = os.path.splitext(path)
        return ext in ['.xml']

    def __get_output_path(self, image, prefix=None, path=None):
        if self.config.keep_file_name and path is not None:
            return path

        md5 = hashlib.md5(image).hexdigest()
        if prefix:
            return '{}/{}@{}.{}'.format(self.output_dir, prefix, md5, self.config.save_image_type)

        return '{}/{}.{}'.format(self.output_dir, md5, self.config.save_image_type)

    def extract_images(self):
        if not self.__is_movie(self.target):
            print('{} is not movie file'.format(self.target))
            return

        cap = cv2.VideoCapture(self.target)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        detector = cv2.AKAZE_create()

        difference = 1000
        _, image = cap.read()
        des_prev = self.__detectAndCompute(image, detector)
        for _ in tqdm(range(frame_count)):
            if not cap.isOpened:
                break

            frame = int(round(cap.get(1)))
            _, image = cap.read()

            if frame % self.config.sampling_frames != 0:
                continue

            des = self.__detectAndCompute(image, detector)
            if des is None:
                continue

            matches = bf.match(des_prev, des)
            dist = [m.distance for m in matches]
            difference = 1000 if len(dist) == 0 else sum(dist) / len(dist)

            if difference > self.config.difference_threshold:
                des_prev = des

                faces = self.detect_faces(image)
                if len(faces) != 0:
                    output_path = self.__get_output_path(image, frame)
                    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])

        cap.release()

    def __detectAndCompute(self, image, detector):
        if image is None:
            return None

        image = cv2.resize(image, (640, 360))
        _, des = detector.detectAndCompute(image, None)
        return des

    def __create_class_path_map(self):
        class_path_map = {}
        for root, dirs, files in os.walk(self.target_dir):
            for file in files:
                if not self.__is_xml(file):
                    continue

                xml_path = os.path.join(root, file)
                with open(xml_path, 'r') as f:
                    xml = ET.fromstring(f.read())

                for obj in xml.findall('object'):
                    cls = obj.find('name').text
                    if cls not in class_path_map:
                        class_path_map[cls] = []

                    name, _ = os.path.splitext(file)
                    class_path_map[cls].append(os.path.join(root, name))

        return class_path_map

    def extract_labels(self):
        for cls, paths in self.__create_class_path_map().items():
            for path in paths:
                print('{} {}'.format(f'{path}.xml', cls))


    def copy_images(self):
        class_path_map = self.__create_class_path_map()
        class_path_map = sorted(class_path_map.items(), key=lambda x:-len(x[1]))

        func = shutil.move if self.config.mv else shutil.copy

        for cls, paths in class_path_map:
            print(cls, len(paths))
            n_sample = min(len(paths), self.config.copy_images_per_class)
            for path in random.sample(paths, n_sample):
                jpg = f'{path}.jpg'
                xml = f'{path}.xml'
                if os.path.exists(jpg): 
                    func(jpg, self.output_dir)
                if os.path.exists(xml): 
                    func(xml, self.output_dir)

    def fix(self):
        for root, dirs, files in os.walk(self.target_dir):
            for file in files:
                if not self.__is_xml(file):
                    continue

                xml_path = os.path.join(root, file)
                with open(xml_path, 'r') as f:
                    xml = ET.fromstring(f.read())

                path = xml.find('path').text
                width = int(xml.find('size/width').text)
                height = int(xml.find('size/height').text)

                writer = Writer(path, height, width, 3)
                for obj in xml.findall('object'):
                    xmin = int(obj.find('bndbox/xmin').text)
                    ymin = int(obj.find('bndbox/ymin').text)
                    xmax = int(obj.find('bndbox/xmax').text)
                    ymax = int(obj.find('bndbox/ymax').text)
                    writer.addObject(obj.find('name').text, xmin, ymin, xmax, ymax)
                writer.save(xml_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='annotate images')
    parser.add_argument('target', default='./images', help='path to target file or directory that has target images')
    parser.add_argument('--mv2img', action='store_true', help='extract images from movie file')
    parser.add_argument('--sampling_frames', type=int, default=30, help='sampling image by this frames')
    parser.add_argument('--difference_threshold', type=int, default=20, help='sampling when image difference is over this value from last sampling image')
    parser.add_argument('--extract_labels', action='store_true', help='extract label from annotation file')
    parser.add_argument('--fix', action='store_true', help='')
    parser.add_argument('--all_faces', action='store_true', help='annotate all faces as "face"')
    parser.add_argument('--recursive', action='store_true', help='search target files recursively')
    parser.add_argument('--output_dir', default=None, help='path to output directory')
    parser.add_argument('--save_image_type', default='jpg', help='image type to save')
    parser.add_argument('--keep_file_name', action='store_true', help='keep original file name to save')
    parser.add_argument('--copy_images_per_class', type=int, default=0, help='copy images with max count per class')
    parser.add_argument('--mv', action='store_true', help='move instead of copy')
    args = parser.parse_args()

    annotator = Annotator(args)

    if args.fix:
        annotator.fix()
        sys.exit()

    if args.mv2img:
        annotator.extract_images()
        sys.exit()

    if args.extract_labels:
        annotator.extract_labels()
        sys.exit()

    if args.copy_images_per_class != 0:
        annotator.copy_images()
        sys.exit()

    if not args.recursive:
        for file in tqdm(os.listdir(args.target)):
            annotator.annotate(os.path.join(args.target, file))
        sys.exit()

    for root, _, files in os.walk(args.target):
        for file in tqdm(files):
            annotator.annotate(os.path.join(root, file))
