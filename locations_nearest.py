import itertools

from get_locations import load_and_preprocess_locations
from utils import CharCountVectorizer, closest_locations_by_vector_distance


if __name__ == "__main__":
    locations = load_and_preprocess_locations()
    char_vectorizer = CharCountVectorizer(1)
    X = char_vectorizer.fit_transform(locations)
    print(list(itertools.islice(closest_locations_by_vector_distance(X, "hechngeli", char_vectorizer, locations), 20)))
    print(list(itertools.islice(closest_locations_by_vector_distance(X, "belir", char_vectorizer, locations), 20)))
