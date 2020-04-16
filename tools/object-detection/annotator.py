import cv2
import sys
import os.path
import argparse
import hashlib
from tqdm import tqdm
from pascal_voc_writer import Writer

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

    def __init__(self, target):
        if os.path.isfile(target):
            self.target = target
            self.target_dir = os.path.dirname(target)
        else:
            self.target = None
            self.target_dir = target

        self.output_dir = os.path.join(self.target_dir, 'output')
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
                                        minNeighbors = 5,
                                        minSize = (24, 24))
    def annotate_image(self, path, image=None):
        if image is None:
            image = cv2.imread(path, cv2.IMREAD_COLOR)

        faces = self.detect_faces(image)

        shape = image.shape
        writer = Writer(path, shape[0], shape[1], shape[2])

        for (x, y, w, h) in faces:
            xmax = x + w
            ymax = y + h
            image_tmp = image.copy()
            cv2.rectangle(image_tmp, (x, y), (xmax, ymax), (0, 0, 255), 2)
            cv2.imshow("annotation", image_tmp)

            label = self.get_label()
            if label is not None:
                writer.addObject(label, x, y, xmax, ymax)
                self.__save_image(image, label)

        writer.save('{}.xml'.format(path))

    def get_label(self):
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

    def __save_image(self, image, label=None):
        md5 = hashlib.md5(image).hexdigest()

        label_name = ''
        if label is not None:
            label_name = self.KEY_LABEL_MAP[label]

        output_path = '{}/{}/{}.jpg'.format(self.output_dir, label_name, md5, 'jpg')
        cv2.imwrite(output_path, image)

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
            if frame % fps != 0:
                continue

            des = self.__detectAndCompute(image, detector)
            if des is None:
                continue

            matches = bf.match(des_prev, des)
            dist = [m.distance for m in matches]
            difference = sum(dist) / len(dist)

            if difference > 20:
                des_prev = des

                faces = self.detect_faces(image)
                if len(faces) != 0:
                    self.__save_image(image)

        cap.release()

    def __detectAndCompute(self, image, detector):
        image = cv2.resize(image, (640, 360))
        _, des = detector.detectAndCompute(image, None)
        return des

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='annotate images')
    parser.add_argument('target', default='./images', help='path to target file or directory that has target images')
    parser.add_argument('--mv2img', action='store_true', help='extract images from movie file')
    args = parser.parse_args()

    annotator = Annotator(args.target)

    if args.mv2img:
        annotator.extract_images()
        sys.exit()

    for file in os.listdir(args.target):
        annotator.annotate(os.path.join(args.target, file))
