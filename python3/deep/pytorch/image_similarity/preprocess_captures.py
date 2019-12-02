import os
import sys
import glob
import shutil
import argparse

def extract_label():
    for file in os.listdir(args.target_dir):
        label = file.rsplit('_', 1)[0]
        print(label)

def classify_with_label():
    assert args.target_label != None, "use --target_label to glob files"
    assert args.dest_dir != None, "use --dest_dir to show directory to put images"

    target = '{}/{}*'.format(args.target_dir, args.target_label)
    for path in glob.glob(target):
        file = os.path.basename(path)
        label = file.rsplit('_', 1)[0]
        label_dir = '{}/{}'.format(args.dest_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        count = len([f for f in os.listdir(label_dir)]) 
        if count < args.count:
            shutil.copyfile(path, '{}/{}'.format(label_dir, file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='check image similarity')
    parser.add_argument('target_dir', help='target image directory path')
    parser.add_argument('--extract_label', action='store_true', help='extract label')
    parser.add_argument('--classify_with_label', action='store_true', help='classify images with label in file name')
    parser.add_argument('--count', type=int, default=10, help='file count to process')
    parser.add_argument('--target_label', default=None, help='label to process')
    parser.add_argument('--dest_dir', default=None, help='destination directory')
    args = parser.parse_args()
    print(args)

    if args.extract_label:
        extract_label()
        sys.exit()

    if args.classify_with_label:
        classify_with_label()
        sys.exit()
