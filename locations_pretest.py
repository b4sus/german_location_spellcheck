import itertools

import numpy as np
from som.bag_of_characters import WordNGramer

from get_locations import load_and_preprocess_locations


def predict(word):
    x_test = word_n_gramer.transform([word])[0]
    d = np.linalg.norm(x_test - X, axis=1)
    while len(d) > 0:
        min_index = np.argmin(d)
        yield locations[min_index]
        d = np.delete(d, min_index)


if __name__ == "__main__":
    locations = load_and_preprocess_locations()
    word_n_gramer = WordNGramer(2)
    word_n_gramer.fit(locations)
    X = word_n_gramer.transform(locations)
    print(list(itertools.islice(predict("hechngeli"), 30)))

