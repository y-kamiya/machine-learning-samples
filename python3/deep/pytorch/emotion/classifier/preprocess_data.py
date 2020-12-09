import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('input_path')
parser.add_argument('output_path')
args = parser.parse_args()


data = pd.read_csv(args.input_path, sep='\t', encoding='utf-8', names=['text', 'label'])

# 不必要なlabelを除外
data = data[data.label != 'neutral']
data = data[data.label != 'empty']

# labelをまとめる
data.label = np.where(data.label == 'enthusiasm', 'joy', data.label)
data.label = np.where(data.label == 'love', 'joy', data.label)
data.label = np.where(data.label == 'fun', 'joy', data.label)
data.label = np.where(data.label == 'relief', 'joy', data.label)
data.label = np.where(data.label == 'happiness', 'joy', data.label)

data.label = np.where(data.label == 'hate', 'disgust', data.label)
data.label = np.where(data.label == 'worry', 'disgust', data.label)

data.label = np.where(data.label == 'boredom', 'sadness', data.label)

data.label = np.where(data.label == 'fear', 'surprise', data.label)

data.to_csv(args.output_path, sep='\t', header=False, index=False)
