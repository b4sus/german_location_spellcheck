from utils import CharCountVectorizer, Suggester
from locations_som import Suggester as SomSuggester

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    suggesters = {"nearest_1g": Suggester(CharCountVectorizer(1)),
                  "nearest_2g": Suggester(CharCountVectorizer(2)),
                  "pca_1g": Suggester(make_pipeline(CharCountVectorizer(1), StandardScaler(), PCA(20))),
                  "pca_2g": Suggester(make_pipeline(CharCountVectorizer(2), StandardScaler(), PCA(20))),
                  "som_1g": SomSuggester("1gram", 10),
                  "som_2g": SomSuggester("2gram", 10)}
    for name, suggester in suggesters.items():
        print(f'{name:10} -> {suggester("hechngeli")}')
