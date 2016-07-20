import sys
import numpy as np
from skimage import io, feature, color
from glob import iglob
import pickle

from calculate_histogram import get_histogram

def get_features(directory):
    features = []
    for fn in iglob('%s/*.png' % directory):
        image = color.rgb2gray(io.imread(fn))
        features.append(get_histogram(image).reshape(-1))
        features.append(get_histogram(np.fliplr(image)).reshape(-1))
    return features

def main():
    positive_dir = sys.argv[1]
    negative_dir = sys.argv[2]
    output_file = sys.argv[3]

    positive_samples = get_features(positive_dir)
    negative_samples = get_features(negative_dir)

    n_positives = len(positive_samples)
    n_negatives = len(negative_samples)

    X = np.array(positive_samples + negative_samples)

    # define labels, positive image:1, negative image:0
    y = np.array([1 for i in range(n_positives)] +
                 [0 for i in range(n_negatives)])

    pickle.dump((X, y), open(output_file, "w"))

if __name__ == "__main__":
    main()
