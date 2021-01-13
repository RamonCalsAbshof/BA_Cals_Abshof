from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors

import numpy as np
import pandas as pd

class OT_DBSCAN(object):

    def __init__(self, eps=0.5, min_samples=5, eps2=None, sw=3, sw_position="center"):
        self.eps = eps
        self.min_samples = min_samples
        if not eps2:
            self.eps2 = eps
        else:
            self.eps2 = eps2
        self.sw = sw
        self.sw_pos = sw_position
        self.len_for_first = self.get_len_for_first()
        self.idx = self.get_index()
        return

    def get_len_for_first(self):
        if self.sw_pos == "prefix":
            len_for_first = 0
        elif self.sw_pos == "center":
            len_for_first = int((self.sw - 1) / 2)
        elif self.sw_pos == "suffix":
            len_for_first = self.sw
        else:
            raise ValueError("sw_position must either be 'prefix', 'center' or 'suffix'.")
        return len_for_first

    def get_index(self):
        if self.sw_pos == "prefix":
            idx = self.sw
        elif self.sw_pos == "center":
            idx = int((self.sw - 1) / 2)
        elif self.sw_pos == "suffix":
            idx = 0
        else:
            raise ValueError("sw_position must either be 'prefix', 'center' or 'suffix'.")
        return idx

    def check_params(self):
        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")
        if not self.min_samples > 0:
            raise ValueError("min_samples must be positive.")
        if not self.eps2 > 0.0:
            raise ValueError("eps2 must be positive.")
        if self.sw_pos not in ["prefix", "center", "suffix"]:
            raise ValueError("sw_position must either be 'prefix', 'center' or 'suffix'.")

    def calc_clusters(self, ot_core_samples, ot_neighbors):
        # Initially, all samples are noise.
        labels = np.full(len(ot_neighbors), -1, dtype=np.intp)

        cur_label = -1

        for core_sample in ot_core_samples:
            if labels[core_sample] > -1:
                cur_label = labels[core_sample]
            else:
                cur_label += 1
                labels[core_sample] = cur_label

            seeds = ot_neighbors[core_sample].copy()
            for neighbor in seeds:
                if int(neighbor) == core_sample:
                    continue
                if labels[neighbor] < 0:
                    labels[neighbor] = cur_label
                    if neighbor in ot_core_samples:
                        seeds.extend(ot_neighbors[neighbor].copy())
                if 48 in seeds:
                    print('now')
        return labels

    def fit(self, X, y=None):
        self.check_params()

        sw_neighbors = []
        eps2_neighbors = []
        ot_neighbors = []
        ot_core_samples = []
        labels = []
        idx = 0
        for time_data in X:
            neighbors_model = NearestNeighbors(radius=self.eps)
            neighbors_model.fit(time_data)
            cur_neighbors = neighbors_model.radius_neighbors(time_data, return_distance=False)
            cur_neighbors = [nbs.astype(int) for nbs in cur_neighbors]

            if self.eps2 != self.eps:
                eps2_neighbors_model = NearestNeighbors(radius=self.eps2)
                eps2_neighbors_model.fit(time_data)
                eps2_neighbors.append(eps2_neighbors_model.radius_neighbors(time_data, return_distance=False))
            else:
                eps2_neighbors.append(cur_neighbors)

            if len(sw_neighbors) < self.len_for_first:
                sw_neighbors.append(cur_neighbors)
            else:
                if len(sw_neighbors) == self.sw:
                    sw_neighbors = sw_neighbors[1:]
                sw_neighbors.append(cur_neighbors)

                ot_neighbors.append(self.get_ot_neighbors(sw_neighbors, eps2_neighbors, idx))

                n_neighbors = np.array([len(neighbors) for neighbors in ot_neighbors[-1]])
                ot_core_samples.append((np.argwhere(n_neighbors >= self.min_samples)).flatten())

                labels.append(self.calc_clusters(ot_core_samples[-1], ot_neighbors[-1]))

                if idx < self.idx:
                    idx += 1

        for last_idx in range(idx, self.sw):
            ot_neighbors.append(self.get_ot_neighbors(sw_neighbors, eps2_neighbors, last_idx))

            n_neighbors = np.array([len(neighbors) for neighbors in ot_neighbors[-1]])
            ot_core_samples.append((np.argwhere(n_neighbors >= self.min_samples)).flatten())

            labels.append(self.calc_clusters(ot_core_samples[-1], ot_neighbors[-1]))

        self.labels_ = labels
        return

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

    def get_ot_neighbors(self, sw_neighbors, eps2_neighbors, idx):
        if self.sw_pos == "prefix":
            prec = idx
            suc = 0

        elif self.sw_pos == "center":
            prec = round((self.sw - 1) / 2)
            if prec > idx:
                prec = idx

            suc = int((self.sw - 1) / 2)
            if suc > len(sw_neighbors) - idx:
                suc = len(sw_neighbors) - idx
        else:
            prec = 0
            suc = len(sw_neighbors) - 1

        ot_neighbors = []
        time_neighbors = sw_neighbors[idx]
        for obj in range(len(time_neighbors)):
            obj_ot_neighbors = set(time_neighbors[obj])
            for i in range(idx - prec, idx + suc + 1):
                if i != idx:
                    obj_ot_neighbors = obj_ot_neighbors.intersection(set(eps2_neighbors[i][obj]))
            ot_neighbors.append(list(obj_ot_neighbors))
        return ot_neighbors


if __name__ == "__main__":
    test_data = pd.DataFrame({'ObjectID': ['a', 'b', 'c', 'd', 'e', 'f',
                                           'a', 'b', 'c', 'd', 'e', 'f',
                                           'a', 'b', 'c', 'd', 'e', 'f',
                                           'a', 'b', 'c', 'd', 'e', 'f',
                                           'a', 'b', 'c', 'd', 'e', 'f'],
                              'Time': [1, 1, 1, 1, 1, 1,
                                       2, 2, 2, 2, 2, 2,
                                       3, 3, 3, 3, 3, 3,
                                       4, 4, 4, 4, 4, 4,
                                       5, 5, 5, 5, 5, 5],
                              'Feature': [0.9, 0.6, 0.5, 0.4, 0.2, 0.1,
                                          0.9, 0.6, 0.5, 0.3, 0.2, 0.1,
                                          0.8, 0.6, 0.5, 0.4, 0.2, 0.1,
                                          0.9, 0.6, 0.5, 0.4, 0.3, 0.1,
                                          0.9, 0.5, 0.6, 0.4, 0.2, 0.1]})
    x_data = []
    timestamps = test_data['Time'].unique()
    timestamps.sort()
    for time in timestamps:
        x_data.append(test_data[test_data['Time'] == time]['Feature'].tolist())
    x_data = np.array(x_data)[:,:,None]
    otdbscan = OT_DBSCAN(eps=0.2, min_samples=3)
    labels = otdbscan.fit_predict(x_data)
    print(labels)