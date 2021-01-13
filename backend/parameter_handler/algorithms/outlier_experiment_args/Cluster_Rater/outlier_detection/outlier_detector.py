import pandas
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from Cluster_Rater.outlier_detection.reference_histogram_outlier import HistOutlier


class OutlierDetector(object):

    def __init__(self, data, weighting=False, jaccard=False, dev_data=None):
        """
        Params:
            data (DataFrame) - pandas DataFrame with columns 'object_id', 'time', 'cluster_id' containing objects,
                               timestamps, cluster belongings, features ..
                               Note: The first three columns can have custom names as long as they represent the object
                               identifier, the timestamp and the cluster identifier in the right order
        Optional:
            weighting (boolean) - indicating whether the weighting function should be applied
            jaccard (boolean) - indicating whether the jaccard index should be used instead of the asymmetric proportion
            dev_data (DataFrame) - pandas DataFrame with columns 'object_id', 'time', 'cluster_id' containing objects,
                                   timestamps, cluster belongings, features .. containing the deviated data
        """
        self._data = data
        self._dev_data = dev_data
        self._weighting = weighting
        self._jaccard = jaccard
        self._outlier_result = None
        self._outlier_rating = None

        self._column_names = data.columns.values
        self._object_column_name = self._column_names[0]
        self._time_column_name = self._column_names[1]
        self._cluster_column_name = self._column_names[2]

        self._cluster_compositions = self.obtain_cluster_compositions()

    def calc_outlier_degree(self):
        """
        Returns:
            outlier_rating (DataFrame) - pandas DataFrame with the columns 'object_id', 'start_time', 'end_time',
                                         'cluster_end_time', 'rating' and 'distance' containing the subsequences'
                                         distances to the reference sequence per time period
        """
        rating = self.calc_outlier_rating()
        outlier = HistOutlier()
        self._outlier_rating = outlier.calc_outlier_degree(rating, self._data)
        return self._outlier_rating

    def mark_outliers(self, tau):
        """
        Parameters:
            tau (float) - threshold for outlier detection

        Returns:
            data (DataFrame) - pandas DataFrame with columns 'object_id', 'time', 'cluster_id', 'outlier'
            outlier_result (DataFrame) - pandas DataFrame with columns 'object_id', 'start_time', 'end_time',
                                         'cluster_end_time', 'rating', 'distance' and 'outlier'
        """
        print('TAU: ', str(tau))
        self._outlier_result = self._outlier_rating[(self._outlier_rating['distance'] >= tau) |
                                                    (self._outlier_rating['distance'] == -1)]

        # mark outliers in the clusters
        self._data = self._data.assign(outlier=1)
        self._data = self._data.astype({self._object_column_name: str})
        self._outlier_result = self._outlier_result.astype({self._object_column_name: str})

        time_points = self._data[self._time_column_name].unique().tolist()
        time_points.sort()

        # mark outliers detected by distance with -1
        for index, row in self._outlier_result.iterrows():
            for time_point in time_points[
                              time_points.index(int(row['start_time'])):time_points.index(int(row['end_time'])) + 1]:
                self._data.loc[(self._data[self._time_column_name] == time_point) &
                               (self._data[self._object_column_name] == row[self._object_column_name]), 'outlier'] = -1

        conseq_outliers = self._outlier_result[self._outlier_result['distance'] == -1]
        # mark conseq cluster outliers with -2, mark conseq outliers which also are outliers by distance with -3
        for index, row in conseq_outliers.iterrows():
            for time_point in time_points[
                              time_points.index(int(row['start_time'])):time_points.index(int(row['end_time'])) + 1]:
                if self._data.loc[(self._data[self._time_column_name] == time_point) &
                                  (self._data[self._object_column_name] == row[self._object_column_name]),
                                  'outlier'].item() in [-1, -2, -3]:
                    self._data.loc[(self._data[self._time_column_name] == time_point) &
                                   (self._data[self._object_column_name] == row[self._object_column_name]),
                                   'outlier'] = -3
                else:
                    self._data.loc[(self._data[self._time_column_name] == int(time_point)) &
                                   (self._data[self._object_column_name] == row[self._object_column_name]),
                                   'outlier'] = -2

        return self._data, self._outlier_result

    def rate_object(self, id=None, start_time=None, end_time=None):
        """
        Optional:
            id (int, str, list) - int, str, list or None representing the data points that should be rated. If id is
                                  None, all objects are rated
            start_time (int) - time that should be considered as beginning
            end_time (int) - int representing the timestamp which should be rated up to

        Returns:
            ratings (dict) - dict {<object_id>: <rating>} with ratings of objects
        """
        ids_to_rate = self.get_ids_to_rate(id, self._object_column_name)
        if end_time is None:
            end_time = np.max(self._data[self._time_column_name].unique())
        ratings = self.calc_object_rating(ids_to_rate, end_time, start_time)
        return ratings

    def calc_object_rating(self, ids_to_rate, end_time, start_time=None):
        """
        Params:
            cluster_composition (dict) - dict {<cluster_id>: {<contained_cluster_id>: <proportion>}} containing the
                                         proportions of clusters (contained_cluster_id) that belong to cluster (cluster_id)
            ids_to_rate (list) - list of data points that should be rated
            end_time (int) - representing the timestamp which should be rated up to
        Optional:
            start_time (int) - time that should be considered as beginning

        Returns:
            ratings (dict) - dict {<object_id>: <rating>} with ratings of objects
        """
        ratings = {}
        gr_clusters = self._data.groupby(self._object_column_name)

        # iterate over object ids
        for id in ids_to_rate:
            cur_group = gr_clusters.get_group(id)
            cur_group = cur_group[cur_group[self._time_column_name] <= end_time]

            if start_time is not None:
                cur_group = cur_group[cur_group[self._time_column_name] >= start_time]

            if len(cur_group[cur_group[self._time_column_name] == end_time][self._cluster_column_name]) == 0:
                # print('Object does not exist for timestamp ', str(end_time))
                continue

            # id of the cluster of the last considered timestamp
            last_cluster = cur_group[cur_group[self._time_column_name] == end_time][self._cluster_column_name].iloc[0]

            # if object is an outlier for the considered timestamp, it gets worst rating of 0.0
            if int(last_cluster) < 0:
                ratings[id] = 0.0
                continue

            cluster_ids = cur_group[self._cluster_column_name].unique()

            object_ratings = []
            num_clusters = 0
            has_outlier = False
            for cluster in cluster_ids:
                if cluster == last_cluster:
                    continue
                # Add the proportion of clusters before last timestamp, that merged in last cluster
                else:
                    # outliers get worst rating of 0.0
                    if int(cluster) < 0:
                        object_ratings.append(0.0)
                        has_outlier = True
                    else:
                        object_ratings.append(self._cluster_compositions[last_cluster][cluster])
                    num_clusters += 1
            if not has_outlier and len(object_ratings) == 0:
                # print(str(id) + " has no data before t=" + str(end_time))
                continue

            if self._weighting:
                try:
                    weighting_denominator = 0
                    for i in range(1, num_clusters + 1):
                        weighting_denominator += i

                    if num_clusters > 0:
                        object_rating = 0
                        for i in range(num_clusters):
                            object_rating += object_ratings[i] * ((i + 1) / weighting_denominator)

                    else:
                        continue
                except (TypeError, ZeroDivisionError):
                    # print(str(id) + " is not assigned to any cluster before t=" + str(end_time))
                    continue
            else:
                try:
                    object_rating = np.sum(object_ratings)
                    object_rating /= num_clusters
                except (TypeError, ZeroDivisionError):
                    # print(str(id) + " is not assigned to any cluster before t=" + str(end_time))
                    continue

            ratings[id] = round(object_rating, 3)
        return ratings

    def calc_outlier_rating(self):
        ratings = []
        timestamps = self._data[self._time_column_name].unique()
        timestamps.sort()
        for i in range(0, len(timestamps) - 1):
            for j in range(i + 1, len(timestamps)):
                time_ratings = self.rate_object(start_time=timestamps[i], end_time=timestamps[j])
                for object in time_ratings:
                    cluster = self._data[(self._data[self._object_column_name] == object) &
                                         (self._data[self._time_column_name] == timestamps[j])
                                         ][self._cluster_column_name].item()
                    ratings.append([object, timestamps[i], timestamps[j], cluster, time_ratings[object]])
        # if self._dev_data is None:
        #     self._dev_data = self.calc_dev_data()
        outlier_rating = pandas.DataFrame(ratings, columns=[self._object_column_name, 'start_time', 'end_time',
                                                            'cluster_end_time', 'rating'])
        # outlier_rating = self.calc_mae(outlier_rating)
        return outlier_rating

    def calc_lof(self, rating):
        ratings = rating.values.tolist()
        rate_values = np.reshape(rating['rating'].values, (-1, 1))
        lof = LocalOutlierFactor()
        predictions = lof.fit_predict(rate_values)
        new_ratings = []
        for row, pred in zip(ratings, predictions):
            new_ratings.append(row + [pred])
        return pandas.DataFrame(new_ratings, columns=[self._object_column_name, 'start_time', 'end_time',
                                                      'cluster_end_time', 'rating', 'lof'])

    def calc_z_score(self, rating):
        ratings = rating.values.tolist()
        rate_values = rating['rating'].values
        mean = np.mean(rate_values)
        std = np.std(rate_values)
        new_ratings = []
        for item in ratings:
            z_score = np.abs((item[4] - mean)) / std
            new_ratings.append(item + [z_score])
        return pandas.DataFrame(new_ratings, columns=[self._object_column_name, 'start_time', 'end_time',
                                                      'cluster_end_time', 'rating', 'z_score'])

    def calc_histogram(self, rating):
        ratings = rating.values.tolist()
        rate_values = rating['rating'].values
        hist = np.histogram(rate_values, bins=20)[0]
        inliner_threshold = 1.0
        for i in range(1, len(hist)):
            if hist[-i] <= 0:
                break
            inliner_threshold = (len(hist) - i) * 0.05
        print(inliner_threshold)
        new_ratings = []
        for item in ratings:
            if inliner_threshold <= item[4] <= 1.0:
                inlier = 1
            else:
                inlier = -1
            new_ratings.append(item + [inlier])
        return pandas.DataFrame(new_ratings, columns=[self._object_column_name, 'start_time', 'end_time',
                                                      'cluster_end_time', 'rating', 'inlier'])

    def calc_mae(self, rating):
        ratings = rating.values.tolist()
        rate_values = rating['rating'].values
        mae = np.mean(1 - rate_values)
        print(mae)
        new_ratings = []
        for item in ratings:
            if item[4] >= 1 - 3 * mae:
                inlier = 1
            else:
                inlier = -1
            new_ratings.append(item + [inlier])
        return pandas.DataFrame(new_ratings, columns=[self._object_column_name, 'start_time', 'end_time',
                                                      'cluster_end_time', 'rating', 'inlier'])

    ######## HELPER FUNCTIONS ########

    def get_feature_list(self, objects, time):
        """
        Params:
            objects (list) - list of objects_ids that belong to considered cluster
            time (int) - time of cluster that is considered

        Returns:
            feature_list (list) - list of lists containing the features of objects in the considered cluster
        """
        feature_list = []
        for obj in objects:
            features = self._data[
                (self._data[self._object_column_name] == obj) & (self._data[self._time_column_name] == time)]
            features = \
                features.drop([self._object_column_name, self._cluster_column_name, self._time_column_name],
                              axis=1).iloc[0].tolist()
            if len(features) <= 0:
                print("No features found for object ", str(obj))
                continue
            feature_list.append(features)
        return np.array(feature_list)

    def get_num_timestamps(self, start_time, end_time):
        """
        Params:
            start_time (int) - first timestamp to be considered
            end_time (int) - last timestamp to be considered

        Returns:
            num_timestamps (int) - number of timestamps between start_time and end_time
        """
        timestamp_list = self._data[self._time_column_name].unique()
        if start_time is not None:
            timestamp_list = [i for i in timestamp_list if i >= start_time]
        if end_time is not None:
            timestamp_list = [i for i in timestamp_list if i <= end_time]
        num_timestamps = len(timestamp_list)
        return num_timestamps

    def get_ids_to_rate(self, id, id_name, start_time=None, end_time=None):
        """
        Params:
            id (int, str, list) - int, str, list or None representing the data points that should be rated. If id is
                                  None, all objects are rated
            id_name (str) - either self._cluster_column_name or self._object_column_name, which ids to extract
        Optional:
            start_time (int) - which timestamp to start at
            end_time (int) - whicht timestamp to stop at

        Returns:
            ids_to_rate (list) - list of ids that should be rated
        """
        if id is None:
            data = self._data.copy()
            if start_time is not None:
                data = data[data[self._time_column_name] >= start_time]
            if end_time is not None:
                data = data[data[self._time_column_name] <= end_time]
            ids_to_rate = data[id_name].unique().tolist()
        elif isinstance(id, int) or isinstance(id, str):
            ids_to_rate = [id]
        elif isinstance(id, list):
            ids_to_rate = id[:]
        else:
            raise Exception('id has to be int, str, list or None')
        return ids_to_rate

    def obtain_cluster_compositions(self):
        """
        Returns:
            cluster_compositions (dict) - dict of dicts {<cluster_id>: {<cluster_id>: <proportion>}} with cluster
                                          compositions

        Example:
            {5: {1: 1.0, 2: 0.1, 4: 0.5}} describes that
                    100% of cluster 1, 10% of cluster 2 and 50% of cluster 4 belong to cluster 5
        """
        cluster_compositions = {}
        g_clusters = self._data.groupby([self._time_column_name, self._cluster_column_name])

        if not self._jaccard:
            cluster_members = self._data.groupby(self._cluster_column_name).count()

        # iterate over all clusters - 'group' contains the time and cluster_id
        # and 'objects' is the corresponding dataframe
        for group, objects in g_clusters:
            # Ignore outliers
            if int(group[1]) < 0:
                continue

            objects = objects[self._object_column_name].values.tolist()

            # temporal intersection
            # select considered clusters with later timestamps than the current one to check which clusters the
            # current one merged into and count, how many objects of the current cluster are in the considered clusters
            # example of a series from the dataframe: [cluster_id, count] with [2, 10]
            # meaning: 10 objects of the current cluster merged into the cluster with the id 2
            temp_intersection = (self._data.loc[(self._data[self._object_column_name].isin(objects)) &
                                                (self._data[self._time_column_name] > group[0])]
                                ).groupby(self._cluster_column_name).count()

            # iterate over all clusters which the current cluster has merged into
            # 'cluster' contains the cluster_id
            # and 'con_objects' is the corresponding number of objects of the temporal intersection
            for cluster, num_objects in temp_intersection.iterrows():
                # Ignore outliers
                if int(cluster) < 0:
                    continue

                # for all considered clusters save the proportion of the current cluster that merged into the considered
                # one
                # example: {3: {2: 0.3}, 4: {2: 0.1}}
                # meaning: 30% of (current) cluster 2 merged into (considered) cluster 3 and 10% into (considered) cluster 4
                if cluster not in cluster_compositions:
                    cluster_compositions[cluster] = {}

                if self._jaccard:
                    # cardinality of the union of both considered clusters
                    card_union = len(self._data.loc[(self._data[self._cluster_column_name] == cluster) |
                                                    (self._data[self._cluster_column_name] == group[1])]
                                     [self._object_column_name].unique())
                    # jaccard distance
                    cluster_compositions[cluster][group[1]] = round(float(num_objects.values[1]) /
                                                                    float(card_union), 3)
                else:
                    cluster_compositions[cluster][group[1]] = round(float(num_objects.values[1]) /
                                                                    float(cluster_members.loc[group[1]].values[1]), 3)
            if group[1] not in cluster_compositions:
                cluster_compositions[group[1]] = {}
        return cluster_compositions

    def calc_dev_data(self):
        columns = self._data.columns
        feature_columns = [i for i in columns if i not in [self._object_column_name, self._time_column_name,
                                                           self._cluster_column_name]]
        objects = self._data[self._object_column_name].unique()
        timestamps = self._data[self._time_column_name].unique()
        new_rows = []
        for object in objects:
            for i in range(len(timestamps) - 1):
                if len(self._data[(self._data[self._object_column_name] == object) &
                                  (self._data[self._time_column_name] == timestamps[i])]) > 0 \
                        and len(self._data[(self._data[self._object_column_name] == object) &
                                           (self._data[self._time_column_name] == timestamps[i + 1])]) > 0:
                    row = self._data[(self._data[self._object_column_name] == object) &
                                     (self._data[self._time_column_name] == timestamps[i])]
                    new_row = row[[self._object_column_name, self._time_column_name, self._cluster_column_name]].iloc[
                        0].tolist()
                    for feature in feature_columns:
                        dev = self._data[(self._data[self._object_column_name] == object) &
                                         (self._data[self._time_column_name] == timestamps[i])][feature].item() - \
                              self._data[(self._data[self._object_column_name] == object) &
                                         (self._data[self._time_column_name] == timestamps[i + 1])][feature].item()
                        new_row.append(dev)
                    new_rows.append(new_row)
        dev_data = pandas.DataFrame(new_rows, columns=columns)
        return dev_data

    def set_dev_data(self, dev_data):
        self._dev_data = dev_data.copy()
        return
