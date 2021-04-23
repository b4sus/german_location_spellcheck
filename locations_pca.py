import itertools

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from get_locations import load_and_preprocess_locations
from utils import CharCountVectorizer, closest_locations_by_vector_distance

if __name__ == "__main__":
    pipeline = make_pipeline(CharCountVectorizer(1), StandardScaler(), PCA(20))

    locations = load_and_preprocess_locations()

    X = pipeline.fit_transform(locations)

    print(list(itertools.islice(closest_locations_by_vector_distance(X, "hechngeli", pipeline, locations), 20)))
