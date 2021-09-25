import argparse
import random
import torch
from transformers import *
from logzero import setup_logger

class Classifier:
    def __init__(self, config):
        self.config = config

        self.tokenizer, self.model = self.__create_model()

        with open(config.filepath, 'r') as f:
            self.sentences = [line.strip() for line in f.readlines()]

    def __create_model(self):
        if self.config.lang == 'ja':
            pretrained_weights = 'cl-tohoku/bert-base-japanese-whole-word-masking'
            tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_weights)
            model = BertForMaskedLM.from_pretrained(pretrained_weights)
            return tokenizer, model

        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        model = BertForMaskedLM.from_pretrained(pretrained_weights)
        return tokenizer, model

    def execute(self):
        for sentence in self.sentences:
            self.classify(sentence)

    def classify(self, sentence):
        input = torch.tensor([self.tokenizer.encode(sentence, add_special_tokens=True)])

        with torch.no_grad():
            output = self.model(input, output_hidden_states=True)

        hidden_state = output[1][1]
        print(hidden_state.shape)
        print(hidden_state[0][1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('filepath', help='file path to target sentences')
    parser.add_argument('--loglevel', default='DEBUG')
    parser.add_argument('--lang', default='en', help='language')
    args = parser.parse_args()

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    detector = Classifier(args)
    detector.execute()

