import base64
import io
import json
import requests
import argparse
import sys
import os
import numpy as np


def container_predict(image_files, image_keys, port_number=8501):
    """Sends a prediction request to TFServing docker container REST API.

    Args:
        image_files: Path to a local image for the prediction request.
        image_keys: Your chosen string key to identify the given image.
        port_number: The port number on your device to accept REST API calls.
    Returns:
        The response of the prediction request.
    """

    instances = { 'instances': [] }

    for i, path in enumerate(image_files):
        key = str(image_keys[i])

        with io.open(path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        instances['instances'].append({
            'image_bytes': {'b64': str(encoded_image)},
            'key': key,
        })

    url = 'http://localhost:{}/v1/models/default:predict'.format(port_number)

    return requests.post(url, data=json.dumps(instances)).json()

def show_top(file, top_labels, top_scores):
    print(os.path.basename(file))
    for i, score in enumerate(top_scores):
        print('{:.4f}, {}'.format(score, top_labels[i]))

def get_top(prediction, top):
    scores = np.array(prediction['scores'])
    labels = np.array(prediction['labels'])

    top_indexes = scores.argsort()[::-1][0:top]
    top_scores = scores[top_indexes]
    top_labels = labels[top_indexes]

    return top_labels, top_scores

def is_ext(path, ext):
    _, str = os.path.splitext(os.path.basename(path))
    return str == ext

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get predicted result from automl edge container on local')
    parser.add_argument('image_file_path', help='image file path')
    parser.add_argument('--top', type=int, default=5, help='show result of top N class')
    parser.add_argument('--synset_labels', default=None, help='file path of labels list')
    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.image_file_path):
        result = container_predict([args.image_file_path], ['test-key'])
        top_labels, top_scores = get_top(result['predictions'][0], args.top)
        show_top(args.image_file_path, top_labels, top_scores)
        sys.exit()

    files = sorted([os.path.join(args.image_file_path, file) for file in os.listdir(args.image_file_path) if is_ext(file, '.jpg')])
    indexes = list(range(len(files)))

    pred_labels = []
    result = container_predict(files, indexes)
    predictions = sorted(result['predictions'], key=lambda x:int(x['key']))

    for i, pred in enumerate(predictions):
        top_labels, top_scores = get_top(pred, args.top)
        pred_labels.append(top_labels)
        show_top(files[i], top_labels, top_scores)

    if args.synset_labels != None:
        num_images = len(pred_labels)
        with open(args.synset_labels, 'r') as f:
            content = f.read()
            true_labels = content.split('\n')
            true_labels.pop(-1)

        assert len(true_labels) == num_images, 'images and labels are mismatched'

        top1_cnt, top5_cnt = 0.0, 0.0
        for i, label in enumerate(true_labels):
          top1_cnt += label in pred_labels[i][:1]
          top5_cnt += label in pred_labels[i][:5]

        top1, top5 = 100 * top1_cnt / num_images, 100 * top5_cnt / num_images
        print('Final: top1_acc = {:4.2f}%  top5_acc = {:4.2f}%'.format(top1, top5))
