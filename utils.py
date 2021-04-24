import itertools
from get_locations import load_and_preprocess_locations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class CharCountVectorizer(CountVectorizer):
    def __init__(self, n_gram):
        super().__init__(analyzer="char", ngram_range=(n_gram, n_gram))

    def transform(self, raw_documents):
        return super().transform(raw_documents).toarray()

    def fit_transform(self, raw_documents, y=None):
        return super().fit_transform(raw_documents, y).toarray()


def suggest_locations(pipeline):
    locations = load_and_preprocess_locations()
    X = pipeline.fit_transform(locations)
    suggestions = None
    while True:
        word = yield suggestions
        suggestions = list(itertools.islice(closest_locations_by_vector_distance(X, word, pipeline, locations), 20))


def closest_locations_by_vector_distance(X, word, transformer, locations):
    x_test = transformer.transform([word])
    d = np.linalg.norm(x_test - X, axis=1)
    for idx in np.argsort(d):
        yield locations[idx]
