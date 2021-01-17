import argparse
import random
import torch
from transformers import *
from logzero import setup_logger

class Predictor:
    def __init__(self, config):
        self.config = config

        self.tokenizer, self.model = self.__create_model()

        with open(config.filepath, 'r') as f:
            self.sentences = [line.strip() for line in f.readlines()]

    def __create_model(self):
        name = 'Helsinki-NLP/opus-mt-en-jap'
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelWithLMHead.from_pretrained(name)
        return tokenizer, model

    def execute(self):
        for sentence in self.sentences:
            self.predict(sentence)

    def predict(self, sentence):
        print(sentence)
        input_ids = self.tokenizer.encode(sentence, return_tensors='pt')
        print(input_ids)
        outputs = self.model.generate(input_ids)
        print(outputs)
        decoded = self.tokenizer.batch_decode(outputs)
        print(decoded)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('filepath', help='file path to target sentences')
    parser.add_argument('--loglevel', default='DEBUG')
    args = parser.parse_args()

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    predictor = Predictor(args)
    predictor.execute()
