import sys
from dataclasses import dataclass, field

import MeCab
import tensorflow_hub as hub
from argparse_dataclass import ArgumentParser
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractorBase:
    def __init__(self, config):
        self.config = config


class FeatureExtractorTfidf(FeatureExtractorBase):
    def __init__(self, config):
        super().__init__(config)

        self.tagger = MeCab.Tagger()
        self.vectorizer = TfidfVectorizer(
            use_idf=True, min_df=0.02, stop_words=[], token_pattern=u"(?u)\\b\\w+\\b"
        )

    def parse(self, text: str) -> str:
        node = self.tagger.parseToNode(text)

        words = []
        while node:
            pos = node.feature.split(",")
            if pos[0] == "動詞":
                words.append(pos[6])
            elif pos[0] != "助詞":
                words.append(node.surface.lower())
            node = node.next

        return " ".join(words)

    def vectorize(self, datapath):
        with open(datapath, "r") as f:
            data = [self.parse(line.strip()) for line in f.readlines()]

        print(data)
        return self.vectorizer.fit_transform(data)


class FeatureExtractorUse(FeatureExtractorBase):
    def __init__(self, config):
        super().__init__(config)
        self.embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
        )

    def vectorize(self, datapath):
        with open(datapath, "r") as f:
            data = [line.strip() for line in f.readlines()]

        return self.embed(data)


@dataclass
class Config:
    datapath: str = field(default="data/test")
    type: str = field(default="use", metadata=dict(choices=["tfidf", "use"]))


if __name__ == "__main__":
    parser = ArgumentParser(Config)
    args = parser.parse_args()

    if args.type == "tfidf":
        extractor = FeatureExtractorTfidf(args)
        vectors = extractor.vectorize(args.datapath)
        print(vectors)
        print(vectors.shape)
        sys.exit()

    if args.type == "use":
        e = FeatureExtractorUse(args)
        vectors = e.vectorize(args.datapath)
        print(vectors)
        print(vectors.shape)
        sys.exit()
