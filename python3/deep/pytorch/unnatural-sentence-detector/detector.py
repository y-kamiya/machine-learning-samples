import argparse
import random
import torch
from transformers import *

class Detector:
    def __init__(self, config):
        self.config = config

        pretrained_weights = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.model = BertForMaskedLM.from_pretrained(pretrained_weights)

        with open(config.filepath, 'r') as f:
            self.sentences = [line.strip() for line in f.readlines()]
            if self.config.random:
                self.sentences = [self.__randomize(s) for s in self.sentences]

    def __randomize(self, sentence):
        words = sentence.split()
        centers = words[1:-1]
        random_words = [words[0]] + random.sample(centers, len(centers)) + [words[-1]]
        return ' '.join(random_words)

    def execute(self):
        for sentence in self.sentences:
            self.detect(sentence)

    def detect(self, sentence):
        list = []
        input_ids = torch.tensor([self.tokenizer.encode(sentence, add_special_tokens=True)])
        n_words = input_ids.shape[1]
        # replace each token except from <CLS>, <SEP> with <MASK>
        for i in range(1, n_words - 1):
            ids = input_ids.clone()
            ids[0][i] = self.tokenizer.mask_token_id
            list.append(ids)

        input = torch.cat(list, dim=0)

        with torch.no_grad():
            output = self.model(input)

        all_scores = output[0]

        print('==========================')
        print(sentence)
        is_strange = False
        total = 0
        for i in range(1, n_words - 1):
            scores = all_scores[i-1][i]
            topk = torch.topk(scores, 5)

            score = Score(input_ids[0][i], scores, self.tokenizer)
            top_scores = [Score(id.item(), scores, self.tokenizer) for id in topk.indices]
            is_strange = is_strange or score.value_std <= 0
            total += score.value_std

            print('original word: {}: top score: {}'.format(score, top_scores))

        print(total / (n_words - 2))
        if is_strange:
            print('this sentence is strange')

class Score:
    def __init__(self, id, scores, tokenizer):
        self.id = id
        self.value = scores[id].item()
        self.word = tokenizer.decode([id])
        min_value = torch.min(scores)
        max_value = torch.max(scores)
        self.value_std = (self.value - min_value) / (max_value - min_value)

    def __repr__(self):
        return '({:.2f}, {})'.format(self.value_std, self.word)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('filepath', help='file path to target sentences')
    parser.add_argument('--random', action='store_true', help='randomize word order')
    args = parser.parse_args()
    print(args)

    detector = Detector(args)
    detector.execute()
