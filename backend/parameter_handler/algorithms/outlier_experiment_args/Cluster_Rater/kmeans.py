from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class K_Means(object):

    KMEANS_SEED = 7

    def __init__(self, data, k):
        self._data = data
        self._k = k

    def create_clusters(self):
        clusters = pd.DataFrame()
        cluster_count = 0
        for timestamp in self._data['Time'].unique():
            timepoint_data = self._data.loc[self._data['Time'] == timestamp]

            cluster_data = [tuple(x) for x in timepoint_data.iloc[:, 2:].values]

            clustering = KMeans(n_clusters=self._k, random_state=self.KMEANS_SEED).fit(cluster_data)

            for i in range(0, len(clustering.labels_)):
                if clustering.labels_[i] > -1:
                    clustering.labels_[i] = clustering.labels_[i] + cluster_count
            cluster_count = max(clustering.labels_) + 1
            idx = np.array(range(timestamp * len(clustering.labels_)
                                 , timestamp * len(clustering.labels_) + len(clustering.labels_)))
            labels_df = pd.DataFrame({'cluster': clustering.labels_, 'idx': idx})
            labels_df = labels_df.set_index('idx')
            timepoint_data.loc[:, 'idx'] = pd.Series(range(timestamp * len(clustering.labels_),
                                                    timestamp * len(clustering.labels_) + len(clustering.labels_)),
                                                    index=timepoint_data.index)
            timepoint_data = timepoint_data.set_index(['idx'])
            clusters = clusters.append(timepoint_data.join(labels_df))

        return clusters