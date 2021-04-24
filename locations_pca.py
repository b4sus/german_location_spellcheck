from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import CharCountVectorizer, suggest_locations

if __name__ == "__main__":
    suggestor = suggest_locations(make_pipeline(CharCountVectorizer(1), StandardScaler(), PCA(20)))
    next(suggestor)

    print(suggestor.send("hechngeli"))
    print(suggestor.send("belir"))
