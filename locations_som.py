import datetime
import itertools
import json
import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from som.som import SOM, TrainedSOM

from get_locations import load_and_preprocess_locations
from utils import CharCountVectorizer


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

    def hyperparams_path(self):
        return self.trained_data_dir + "/hyperparams.npy"


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


def suggest_locations(config_key):
    current_config = CONFIGS[config_key]
    current_config.ensure_dir()
    if not current_config.trained():
        hyperparams = {"map_length": 100, "learning_rate_constant": 40_000, "init_sigma": 25, "sigma_constant": 35_000,
                       "max_iter": 100_000}
        train(current_config, hyperparams, Observer(draw=False))

    map, map_length, preprocess_pipeline = prepare_trained_pipeline(current_config)
    # map is now map_length x map_length grid of lists of locations

    suggestions = None
    while True:
        word = yield suggestions
        sample_coords = preprocess_pipeline.transform([word])[0]
        generator = generate_closest(sample_coords, map_length, map)
        suggestions = list(itertools.islice(generator, 20))


def generate_closest(sample_bmu_coords, map_length, location_map):
    map_coords = np.empty((map_length, map_length, 2))
    for neuron_x in range(map_length):
        for neuron_y in range(map_length):
            map_coords[neuron_x, neuron_y] = np.array([neuron_x, neuron_y])

    # map_coords are just coordinates in the map like [[[0 0] [0 1] [0 2]]]

    D = np.linalg.norm(sample_bmu_coords - map_coords, axis=2)
    # D contains distance of each neuron in a map to sample bmu coords

    xs, ys = np.unravel_index(np.argsort(D, axis=None), D.shape)
    for x, y in zip(xs, ys):
        if location_map[x, y] is not None:
            for loc in location_map[x, y]:
                yield loc


def train(current_config, hyperparams, observer):
    som = SOM(**hyperparams, observer=observer)
    pipeline = make_pipeline(CharCountVectorizer(current_config.n_gram_param), MinMaxScaler(), som)
    pipeline.fit_transform(LOCATIONS, som__X_repr=LOCATIONS)

    save_after_training(current_config, hyperparams, som)


def save_after_training(current_config, hyperparams, som):
    np.save(current_config.trained_thetas_path(), som.Thetas)
    with open(current_config.hyperparams_path(), mode="w", encoding="utf-8") as fp:
        json.dump(hyperparams, fp)


def prepare_trained_pipeline(current_config):
    Thetas = np.load(current_config.trained_thetas_path())
    trained_map = TrainedSOM(Thetas)
    preprocess_pipeline = make_pipeline(CharCountVectorizer(current_config.n_gram_param), MinMaxScaler(), trained_map)
    preprocess_pipeline.fit(LOCATIONS)
    if current_config.positions_prepared():
        locations_bmu_coords = np.load(current_config.location_positions_path(), allow_pickle=True)
    else:
        locations_bmu_coords = preprocess_pipeline.transform(LOCATIONS)
        np.save(current_config.location_positions_path(), locations_bmu_coords)
    map_length = len(Thetas)
    map = np.empty((map_length, map_length), dtype=object)
    for location, bmu_coords in zip(LOCATIONS, locations_bmu_coords):
        x, y = bmu_coords
        if map[x, y] is None:
            map[x, y] = []
        map[x, y].append(location)
    # map is now map_length x map_length grid of lists of locations
    return map, map_length, preprocess_pipeline


if __name__ == "__main__":
    suggestor = suggest_locations("1gram")
    next(suggestor)
    print(suggestor.send("hechngeli"))
