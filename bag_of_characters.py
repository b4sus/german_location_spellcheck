import numpy as np


class WordNGramer:
    def __init__(self, n):
        self.n = n

    def fit(self, words, *args):
        corpus = set()
        for word in words:
            for n_gram in self.split(word):
                corpus.add(n_gram)
        self.corpus = list(sorted(corpus))
        return self

    def transform(self, words):
        X = np.zeros((len(words), len(self.corpus)), dtype=np.int32)
        for word_idx, word in enumerate(words):
            for n_gram_idx, n_gram in enumerate(self.corpus):
                X[word_idx, n_gram_idx] = word.count(n_gram)
        return X

    def split(self, word):
        for i in range(0, len(word) - 1):
            yield word[i:i + self.n]
