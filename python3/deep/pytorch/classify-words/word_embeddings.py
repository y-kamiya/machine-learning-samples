import sys
import argparse
import heapq
import random
import numpy as np
from logzero import setup_logger
from wikipedia2vec import Wikipedia2Vec

class SimilarWordDetector():
    CATEGORIES = ['ライブ','ポップ','カルチャー','社会','科学','技術','総合', '知識']

    def __init__(self, pkl_path, config):
        self.model = Wikipedia2Vec.load(pkl_path)
        self.config = config

    def execute(self):
        for word in self.CATEGORIES:
            print('original word: {}'.format(word))
            similar_words = self.similar_words(word)
            for entry in similar_words:
                print('{}\t{}'.format(entry[0], entry[1]))

    def test(self):
        print(self.model.dictionary.__dict__)
        # words = self.model.dictionary.words()
        # for i, word in enumerate(words):
        #     if i == 10:
        #         break
        #     print(word)


    def similar_words(self, original_word):
        model_dim = self.model.train_params['dim_size']
        n_top = self.config.n_top
        n_dim = self.config.n_dim
        if model_dim <= n_dim:
            return self.model.most_similar(self.model.get_word(original_word), n_top)

        sampling_dims = random.sample(range(0, model_dim), n_dim)
        original_vector = self.model.get_word_vector(original_word)
        original_vector = self.filter_vector(original_vector, sampling_dims)

        queue = []
        for word in self.model.dictionary.words():
            vector = self.model.get_word_vector(word.text)
            vector = self.filter_vector(vector, sampling_dims)
            similarity = self.cos_sim(original_vector, vector)
            heapq.heappush(queue, (-similarity, word))

        words = []
        for i in range(n_top):
            similarity, word = heapq.heappop(queue)
            words.append((word, -similarity))
        
        return words
        
    def filter_vector(self, vector, indexes):
        filter = np.zeros(self.model.train_params['dim_size'])
        for index in indexes:
            filter[index] = 1

        return vector * filter

    def cos_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('pkl_path', help='pretrained pkl file path')
    parser.add_argument('--loglevel', default='DEBUG')
    parser.add_argument('--n_dim', type=int, default=300, help='dimention to calculate similarity')
    parser.add_argument('--n_top', type=int, default=10, help='number of output words')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    detector = SimilarWordDetector(args.pkl_path, args)
    if args.test:
        detector.test()
        sys.exit()

    detector.execute()
