import cv2
import sys
import os.path
import argparse
from pascal_voc_writer import Writer

script_dir = os.path.dirname(os.path.realpath(__file__))

class Annotator():

    CASCADE_FILE = os.path.join(script_dir, 'lbpcascade_animeface.xml')
    KEY_LABEL_MAP = {
        '0': 'asuna',
        '1': 'administrator',
        '2': 'yuki',
    }

    def annotate(self, image_path):
        if not self.__is_image(image_path):
            return

        if not os.path.isfile(self.CASCADE_FILE):
            raise RuntimeError("%s: not found" % self.CASCADE_FILE)

        cascade = cv2.CascadeClassifier(self.CASCADE_FILE)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        shape = image.shape
        writer = Writer(image_path, shape[0], shape[1], shape[2])

        faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor = 1.1,
                                         minNeighbors = 5,
                                         minSize = (24, 24))
        for (x, y, w, h) in faces:
            xmax = x + w
            ymax = y + h
            image_tmp = image.copy()
            cv2.rectangle(image_tmp, (x, y), (xmax, ymax), (0, 0, 255), 2)
            cv2.imshow("annotation", image_tmp)

            label = self.get_label()
            if label is not None:
                writer.addObject(label, x, y, xmax, ymax)

        writer.save('{}.xml'.format(image_path))

    def get_label(self):
        key = chr(cv2.waitKey(0))
        if key not in self.KEY_LABEL_MAP:
            return None

        return self.KEY_LABEL_MAP[key]

    def __is_image(self, path):
        _, ext = os.path.splitext(path)
        return ext in ['.jpg', '.jpeg', '.png']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='annotate images')
    parser.add_argument('image_dir', default='./images', help='path to directory that has target images')
    args = parser.parse_args()

    annotator = Annotator()

    for file in os.listdir(args.image_dir):
        annotator.annotate(os.path.join(args.image_dir, file))
