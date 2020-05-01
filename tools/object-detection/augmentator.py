import os
import cv2
import hashlib
import argparse
import albumentations as alb
from tqdm import tqdm
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET

class Augmentator():
    def __init__(self):
        pass

    def execute(self, path):
        if not self.__is_xml(path):
            return

        name, _ = os.path.splitext(path)
        image = cv2.imread('{}.jpg'.format(name))

        annotations = self.__build_annotations(image, path)

        flip_dir = os.path.join(os.path.dirname(path), 'flip')
        self.__save_result([alb.HorizontalFlip(p=1)], annotations, flip_dir)

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
        resized_ratio = [0.125, 0.25, 0.5, 1, 2, 4, 8]

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

        cv2.imwrite(output_path, result['image'])

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
    args = parser.parse_args()

    augmentator = Augmentator()

    for file in tqdm(os.listdir(args.target)):
        augmentator.execute(os.path.join(args.target, file))


# def get_aug(aug, min_area=0., min_visibility=0.):
#     return alb.Compose(aug, bbox_params=alb.BboxParams(format='coco', min_area=min_area, 
#                                                min_visibility=min_visibility, label_fields=['category_id']))
#
# BOX_COLOR = (255, 0, 0)
# TEXT_COLOR = (255, 255, 255)
#
# def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
#     x_min, y_min, w, h = bbox
#     x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
#     class_name = class_idx_to_name[class_id]
#     ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
#     cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
#     cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
#     return img
#
# def visualize(annotations, category_id_to_name):
#     img = annotations['image'].copy()
#     for idx, bbox in enumerate(annotations['bboxes']):
#         img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
#     plt.figure(figsize=(12, 12))
#     plt.imshow(img)
#
# image = cv2.imread('/Users/yuji.kamiya//Desktop/test/output/08213b6332f0439471b12d762964d19d.jpg')
# bboxes = [[13, 17, 60, 60]]
# annotations = {'image': image, 'bboxes': bboxes, 'category_id':[1,2]}
# visualize(annotations, {1:'aaa', 2:'bbb'})
#
# aug = get_aug([alb.HorizontalFlip(p=1)])
# augmented = aug(**annotations)
# print(augmented['bboxes'])
#
# visualize(augmented, {1:'aaa', 2:'bbb'})
