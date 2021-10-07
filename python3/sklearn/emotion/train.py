import MeCab
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor():
    def __init__(self):
        self.tagger = MeCab.Tagger()
        self.vectorizer = TfidfVectorizer(use_idf=True, min_df=0.02, stop_words=[],token_pattern=u'(?u)\\b\\w+\\b')

    def parse(self, text: str) -> str:
        node = self.tagger.parseToNode("野球をします")

        words = []
        while node:
            pos = node.feature.split(",")
            if pos[0] in ["名詞", "形容詞"]:
                words.append(node.surface.lower())
            elif pos[0] == "動詞":
                words.append(pos[6])
            node = node.next

        return " ".join(words)

    def vectorize(self, file_path):
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        data = self.parse(lines)
        return self.vectorizer.fit_trasform(data)


if __name__ == "__main__":
    extractor = FeatureExtractor()
    vectors = extractor.vectorize("./data/test")
    print(vectors)

