import datetime
import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
import som.bag_of_characters as boc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from som.som import SOM, TrainedSOM

from get_locations import load_and_preprocess_locations


class Config:

    def __init__(self, n_gram_param, trained_data_dir):
        self.n_gram_param = n_gram_param
        self.trained_data_dir = trained_data_dir

    def ensure_dir(self):
        try:
            os.mkdir(self.trained_data_dir)
        except FileExistsError:
            pass

    def trained(self):
        return os.path.exists(self.trained_thetas_path())

    def positions_prepared(self):
        return os.path.exists(self.location_positions_path())

    def trained_thetas_path(self):
        return self.trained_data_dir + "/trained_thetas.npy"

    def location_positions_path(self):
        return self.trained_data_dir + "/location_positions.npy"


CONFIGS = {"1gram": Config(1, "data/locations_1gram"),
           "2gram": Config(2, "data/locations_2gram"),
           "2gram2": Config(2, "data/locations_2gram2"),
           "3gram": Config(3, "data/locations_3gram")}

LOCATIONS = load_and_preprocess_locations()


class Observer:
    def __init__(self, draw):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.start = time.process_time()
        self.elapsed_times = []
        self.draw = draw

    def __call__(self, X, map_length, iter, Thetas, learning_rate, sigma, X_repr, neighbor_change=-1):
        self.elapsed_times.append(time.process_time() - self.start)
        if len(self.elapsed_times) % 10 == 0:
            print(f"{datetime.datetime.now()}: iteration {iter}, learning_rate:{learning_rate}, sigma: {sigma}, "
                  f"neighbour_change: {neighbor_change} mean for one training sample: {np.mean(self.elapsed_times)}")
            self.elapsed_times.clear()
        if iter in (100, 1000, 10000):
            self.draw_som(Thetas, X, X_repr, iter, learning_rate, map_length, neighbor_change, sigma)
        self.start = time.process_time()

    def draw_som(self, Thetas, X, X_repr, iter, learning_rate, map_length, neighbor_change, sigma):
        map = np.empty((map_length, map_length), dtype=object)
        for i, x in enumerate(X):
            bmu_coords = np.unravel_index(np.argmin(np.linalg.norm(Thetas - x, axis=2)), (map_length, map_length))
            if map[bmu_coords[0], bmu_coords[1]] is None:
                map[bmu_coords[0], bmu_coords[1]] = []
            map[bmu_coords[0], bmu_coords[1]].append(X_repr[i])
        plt.title(
            f"Iteration {iter} - learning rate {learning_rate:.3f}, sigma {sigma:.3f}, "
            + f"neighbour_change {neighbor_change:.3f}")
        for neuron_x in range(map_length):
            for neuron_y in range(map_length):
                items = map[neuron_x, neuron_y]
                if items:
                    self.ax.text(neuron_x / map_length, neuron_y / map_length,
                                 str(len(items)) + " " + "\n".join(items[:3]),
                                 fontsize='xx-small')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.ax.clear()


def predict(word):
    Thetas = np.load(CURRENT_CONFIG.trained_thetas_path())
    trained_map = TrainedSOM(Thetas)

    preprocess_pipeline = make_pipeline(boc.WordNGramer(CURRENT_CONFIG.n_gram_param), MinMaxScaler(), trained_map)
    preprocess_pipeline.fit(LOCATIONS)

    if CURRENT_CONFIG.positions_prepared():
        location_positions = np.load(CURRENT_CONFIG.location_positions_path(), allow_pickle=True)
    else:
        location_positions = preprocess_pipeline.transform(LOCATIONS)
        np.save(CURRENT_CONFIG.location_positions_path(), location_positions)

    map_length = len(Thetas)
    coords = np.empty((map_length, map_length, 2))
    for neuron_x in range(map_length):
        for neuron_y in range(map_length):
            coords[neuron_x, neuron_y] = np.array([neuron_x, neuron_y])

    map = np.empty((map_length, map_length), dtype=object)
    for location, position in zip(LOCATIONS, location_positions):
        x, y = position
        if map[x, y] is None:
            map[x, y] = []
        map[x, y].append(location)

    transformed_samples = preprocess_pipeline.transform([word])

    generator = generate_closest(transformed_samples[0], coords, map)

    for i in range(30):
        print(next(generator))


def generate_closest(sample, coords, location_map):
    start = time.process_time()
    D = np.linalg.norm(sample - coords, axis=2)
    xs, ys = np.unravel_index(np.argsort(D, axis=None), D.shape)
    print(time.process_time() - start)
    for x, y in zip(xs, ys):
        if location_map[x, y] is not None:
            for loc in location_map[x, y]:
                yield loc


def train(hyperparams, observer):
    map = SOM(**hyperparams, observer=observer)
    pipeline = make_pipeline(boc.WordNGramer(CURRENT_CONFIG.n_gram_param), MinMaxScaler(), map)
    pipeline.fit_transform(LOCATIONS, som__X_repr=LOCATIONS)
    np.save(CURRENT_CONFIG.trained_thetas_path(), map.Thetas)


if __name__ == "__main__":
    CURRENT_CONFIG = CONFIGS["1gram"]
    CURRENT_CONFIG.ensure_dir()
    if not CURRENT_CONFIG.trained():
        hyperparams = {"map_length": 100, "learning_rate_constant": 40_000, "init_sigma": 25, "sigma_constant": 35_000,
                       "max_iter": 10_000}
        train(hyperparams, Observer(draw=False))
    predict("hechngeli")
