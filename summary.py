from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from locations_som import Suggester as SomSuggester
from utils import CharCountVectorizer, Suggester

if __name__ == "__main__":
    suggesters = {"nearest_1g": Suggester(CharCountVectorizer(1)),
                  "nearest_2g": Suggester(CharCountVectorizer(2)),
                  "nearest_12g": Suggester(CharCountVectorizer(1, 2)),
                  "pca_1g": Suggester(make_pipeline(CharCountVectorizer(1), StandardScaler(), PCA(10))),
                  "pca_2g": Suggester(make_pipeline(CharCountVectorizer(2), StandardScaler(), PCA(20))),
                  "pca_12g": Suggester(make_pipeline(CharCountVectorizer(1, 2), StandardScaler(), PCA(20))),
                  "som_1g": SomSuggester("1gram", 10),
                  "som_2g": SomSuggester("2gram", 10)
                  }
    words_to_try = ["hechngeli", "belir", "stuttgart", "stutgart"]
    print("|word|suggester|suggestions|")
    print("|----------|----------|------------------------------------------------------------------------------------"
          "------------------------------------------------------|")
    for word, suggester_name, suggested_values in ((word, name, suggester(word)) for word in words_to_try for
                                                   name, suggester in suggesters.items()):
        # print(f'{word:10} by {suggester_name:10} -> {suggested_values}')
        print(f'|{word}|{suggester_name}|{suggested_values}|')
