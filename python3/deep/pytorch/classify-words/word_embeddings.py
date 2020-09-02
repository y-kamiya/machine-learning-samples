import sys
import argparse
from logzero import setup_logger
from wikipedia2vec import Wikipedia2Vec

class WordEmbeddings():
    CATEGORIES = ['ライブ','ポップ','カルチャー','社会','科学','技術','総合', '知識']

    def __init__(self, pkl_path, config):
        self.model = Wikipedia2Vec.load(pkl_path)
        self.config = config

    def execute(self):
        for word in self.CATEGORIES:
            print('original word: {}'.format(word))
            similar_words = self.model.most_similar(self.model.get_word(word), 200)
            for entry in similar_words:
                print('{}\t{}'.format(entry[0], entry[1]))



        # words = self.model.dictionary.words()
        # for i, word in enumerate(words):
        #     if i == 10:
        #         break
        #     print(word)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('pkl_path', help='pretrained pkl file path')
    parser.add_argument('--loglevel', default='DEBUG')
    args = parser.parse_args()

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    word_embeddings = WordEmbeddings(args.pkl_path, args)
    word_embeddings.execute()
