import sys
import os
import argparse
import heapq
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from logzero import setup_logger
from wikipedia2vec import Wikipedia2Vec

class WordDataset(Dataset):
    def __init__(self, model):
        self.model = model

    def __getitem__(self, index):
        word = self.model.dictionary.get_word_by_index(index)
        vector = self.model.get_vector(word)
        return {'text': word.text, 'vector': torch.tensor(vector, dtype=torch.float32)}

    def __len__(self):
        return self.model.dictionary.word_size

class SimilarWordDetector():
    def __init__(self, pkl_path, config):
        self.model = Wikipedia2Vec.load(pkl_path)
        self.config = config

    def execute(self):
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        for word in self.config.search_words:
            print('original word: {}'.format(word))
            similar_words = self.similar_words(word)
            if similar_words is None:
                continue

            with open('{}/{}.txt'.format(output_dir, word), 'w') as f:
                for entry in similar_words:
                    f.write('{}\t{:.3f}\n'.format(entry[0], entry[1]))

    def similar_words(self, original_word):
        try:
            original_vector = self.model.get_word_vector(original_word)
        except KeyError:
            print('{} is not included in word2vec'.format(original_word))
            return None

        model_dim = self.model.train_params['dim_size']
        n_top = self.config.n_top
        n_dim = self.config.n_dim

        if model_dim <= n_dim:
            return self.model.most_similar(self.model.get_word(original_word), n_top)

        sampling_dims = random.sample(range(0, model_dim), n_dim)
        original_vector = torch.tensor([original_vector], dtype=torch.float32, device=self.config.device)
        original_vector = self.filter_vector(original_vector, sampling_dims)

        dataset = WordDataset(self.model)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        all_similarities = torch.empty(0)
        all_texts = []
        for i, data in enumerate(dataloader):
            texts = data['text']
            vectors = data['vector'].to(self.config.device)
            vectors = self.filter_vector(vectors, sampling_dims)
            similarities = torch.cosine_similarity(vectors, original_vector)
            all_similarities = torch.cat((all_similarities, similarities.cpu()))
            all_texts.extend(texts)

        topk = torch.topk(all_similarities, n_top)

        data = []
        for similarity, index in zip(topk[0], topk[1]):
            data.append((all_texts[index], similarity))
        
        return data
        
    def filter_vector(self, tensor, indexes):
        filter = torch.zeros(self.model.train_params['dim_size'], device=self.config.device)
        for index in indexes:
            filter[index] = 1

        return tensor * filter

    def cos_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def collect_all_words(self):
        for word in self.model.dictionary.words():
            print(word.text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('pkl_path', help='pretrained pkl file path')
    parser.add_argument('--loglevel', default='DEBUG')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--n_dim', type=int, default=300, help='dimention to calculate similarity')
    parser.add_argument('--n_top', type=int, default=10, help='number of output words')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--search_words', default=[], nargs='*')
    parser.add_argument('--collect_all_words', action='store_true')
    args = parser.parse_args()

    is_cpu = args.cpu or not torch.cuda.is_available()
    args.device_name = "cpu" if is_cpu else "cuda"
    args.device = torch.device(args.device_name)

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    detector = SimilarWordDetector(args.pkl_path, args)
    if args.collect_all_words:
        detector.collect_all_words()
        sys.exit()

    detector.execute()
