import pandas as pd
import numpy as np
from Cluster_Rater.rand_index import rand_score
from scipy.spatial.distance import euclidean


class CLOSEv2(object):


    '''
    Input:
        data - pandas dataframe with columns 'object_id', 'time', 'cluster_id' containing cluster belongings, features ..
               Note: outliers should have negative labels/cluster_ids, these should be different for different times
        measure (optional) - str or callable for used quality measure
        minPts (optional) - int: used minPts for density-based quality measure
    '''
    def __init__(self, data, measure='mse', minPts=None, output=False):
        self._data = data
        self._column_names = data.columns.values
        self._object_column_name = self._column_names[0]
        self._time_column_name = self._column_names[1]
        self._cluster_column_name = self._column_names[2]
        self._object_ids = self._data[self._object_column_name].unique()
        self._num_objects = len(self._object_ids)
        self._memberships = self.calc_membership_matrices()

        self._minPts = minPts
        self._output = output
        self.pos_measures = {'sse': self.calc_sse,  # NOTE: sse is not between 0 and 1
                             'mse': self.calc_mse,
                             'mae': self.calc_mae,
                             'max': self.calc_max_dist,
                             'dbi': self.calc_min_pts}

        if measure in self.pos_measures:
            self.measure = self.pos_measures[measure]
        elif callable(measure):
            self.measure = measure
        else:
            self.measure = self.pos_measures['mse']


    def calc_membership_matrices(self):
        memberships = {}
        timestamps = self._data[self._time_column_name].unique()

        for i in range(len(timestamps)):
            relevant_data = self._data[self._data[self._time_column_name] == timestamps[i]]
            memberships[timestamps[i]] = np.zeros((self._num_objects, self._num_objects))
            cluster_ids = relevant_data[self._cluster_column_name].unique()
            for cid in cluster_ids:
                if cid >= 0:
                    members = relevant_data[relevant_data[self._cluster_column_name] == cid][self._object_column_name].unique()
                    for j in range(len(members)-1):
                        index_j = np.argwhere(self._object_ids == members[j])[0][0]
                        for k in range(j+1, len(members)):
                            index_k = np.argwhere(self._object_ids == members[k])[0][0]
                            memberships[timestamps[i]][index_j][index_k] = 1
                            memberships[timestamps[i]][index_k][index_j] = 1
        return memberships


    def calc_cohesion_matrix(self, start_time, end_time):
        timestamps = self._data[self._time_column_name].unique()
        timestamps = timestamps[np.where(timestamps >= start_time)]
        timestamps = timestamps[np.where(timestamps <= end_time)]
        cohesion_matrix = np.zeros((self._num_objects, self._num_objects))
        for time in timestamps:
            cohesion_matrix = cohesion_matrix + self._memberships[time]
        return cohesion_matrix


    def calc_subsequence_ratings(self, start_time, end_time):
        subsequence_ratings = {}
        cohesion_matrix = self.calc_cohesion_matrix(start_time, end_time)

        for i in range(self._num_objects):
            summed = np.sum(cohesion_matrix[i])
            non_zero = np.count_nonzero(cohesion_matrix[i])
            num_timestamps = len(self._data[(self._data[self._object_column_name] == self._object_ids[i]) &
                                            (self._data[self._time_column_name] >= start_time) &
                                            (self._data[self._time_column_name] <= end_time)])

            if non_zero > 0 and num_timestamps > 1:
                rating = summed / non_zero
                rating = rating / num_timestamps
                subsequence_ratings[str(self._object_ids[i])] = rating

        return subsequence_ratings

    '''     
    Input:
        start_time (optional) - int: time that should be considered as beginning
        end_time (optional) - int: time which should be rated up to 
        return_measures (optional) - boolean: whether additional information such as average stability 
                                              and quality should be returned 
    Output:
        CLOSE score - float with rating of clustering
                      dict with additional information if 'return_measures' is True
    '''
    def rate_clustering(self, start_time=None, end_time=None, return_measures=False):
        timestamps = self._data[self._time_column_name].unique()
        if start_time is None:
            start_time = np.min(timestamps)
        if end_time is None:
            end_time = np.max(timestamps)

        cluster_ratings = self.calc_cluster_rating(start_time, end_time)
        gr_clusters = self._data.groupby(self._cluster_column_name)

        score = 0
        avg_quality = 0
        avg_stab = 0

        for cluster in cluster_ratings:
            cluster_objects = gr_clusters.get_group(cluster)[self._object_column_name].unique()
            cluster_time = gr_clusters.get_group(cluster)[self._time_column_name].iloc[0]
            feature_list = self.get_feature_list(cluster_objects, cluster_time)

            measure = self.measure(feature_list)
            avg_quality += measure
            avg_stab += cluster_ratings[cluster]
            score += (cluster_ratings[cluster] * (1 - measure))

        num_clusters = len(cluster_ratings)
        num_timestamps = len(self._data[(self._data[self._time_column_name] >= start_time) &
                                        (self._data[self._time_column_name] <= end_time)][self._time_column_name].unique())

        if num_clusters <= 0:
            if self._output:
             print('Clustering has no Clusters!!')
            return 0

        avg_quality /= num_clusters
        if self._output:
            print('Average Quality: ', str(avg_quality))
        avg_stab /= num_clusters
        if self._output:
            print('Average Stability: ', str(avg_stab))

        factor = (1 / num_clusters) * (1 - (num_timestamps / num_clusters)**2)

        if not return_measures:
            return score * factor

        else:
            return {'close': score * factor,
                    'stability': avg_stab,
                    'quality': avg_quality,
                    'pre-factor': (1 - (num_timestamps / num_clusters) ** 2)}

    '''
    Input:
        ids_to_rate - list of clusters that should be rated
        start_time (optional) - int: time that should be considered as beginning

    Output:
        ratings - dict {<cluster_id>: <rating>} with ratings of clusters
    '''

    def calc_cluster_rating(self, start_time=None, end_time=None):
        timestamps = self._data[self._time_column_name].unique()
        timestamps.sort()
        if start_time is None:
            start_time = timestamps[0]
        if end_time is None:
            end_time = timestamps[-1]

        timestamps = [x for x in timestamps if start_time <= x <= end_time]

        relevant_data = self._data[(self._data[self._time_column_name] >= start_time) &
                                   (self._data[self._time_column_name] <= end_time)]

        ratings = {}

        for time in timestamps:
            cluster_ids = relevant_data[(relevant_data[self._cluster_column_name] >= 0) &
                                     (relevant_data[self._time_column_name] == time)][self._cluster_column_name].unique()

            if time == start_time:
                for id in cluster_ids:
                    ratings[id] = 1.0
                continue

            object_ratings = self.calc_subsequence_ratings(start_time, time)

            for id in cluster_ids:
                obj_list = relevant_data[relevant_data[self._cluster_column_name] == id][self._object_column_name].unique()

                score = 0
                num_objects = 0
                for obj in obj_list:
                    if str(obj) in object_ratings:
                        score += object_ratings[str(obj)]
                        num_objects += 1
                try:
                    score /= num_objects
                except ZeroDivisionError:
                    if self._output:
                        print('Cluster ', str(id), ' has no non-outlier members.')
                    else:
                        continue

                num_merged_clusters = len(relevant_data[(relevant_data[self._time_column_name] < time) &
                                                        (relevant_data[self._object_column_name].isin(obj_list)) &
                                                        (relevant_data[self._cluster_column_name] >= 0)][self._cluster_column_name])
                num_timestamps = len([x for x in timestamps if start_time <= x <= time])
                try:
                    div = num_merged_clusters / num_timestamps
                    score /= div
                except ZeroDivisionError:
                    if self._output:
                        print("<<ZeroDivisionError - Cluster Score>> Cluster ID: ", str(id), "  Merged Clusters: ", str(num_merged_clusters),
                         "  Num Timestamps: ", str(num_timestamps))
                    else:
                        continue
                ratings[id] = score
        return ratings


    '''
    Input:
        objects - list of objects_ids that belong to considered cluster
        time - time of cluster that is considered

    Output:
        feature_list - list of lists containing the features of objects in the considered cluster
    '''
    def get_feature_list(self, objects, time):
        feature_list = []
        for obj in objects:
            features = self._data[
                (self._data[self._object_column_name] == obj) & (self._data[self._time_column_name] == time)]
            try:
                features = \
                    features.drop([self._object_column_name, self._cluster_column_name, self._time_column_name],
                                  axis=1).iloc[0].tolist()
            except IndexError:
                print(">>INDEXERROR - FEATURE LIST<< ID: ", str(obj), ", Time: ", str(time))
                continue

            if len(features) <= 0:
                print("No features found for object ", str(obj))
                continue
            feature_list.append(features)
        return np.array(feature_list)


    ######## QUALITY MEASURES ########

    '''
        Input:
            feature_list - list of lists containing the features of objects in the considered cluster

        Output:
            sse - float: sum of squared errors to centroid of cluster
    '''
    def calc_sse(self, feature_list):
        centroid = np.average(feature_list, axis=0)
        sse = np.sum(np.power(feature_list - centroid[None, :], 2))
        return sse

    '''
        Input:
            feature_list - list of lists containing the features of objects in the considered cluster

        Output:
            mse - float: mean squared error of cluster
    '''
    def calc_mse(self, feature_list):
        sse = self.calc_sse(feature_list)
        return sse / len(feature_list)

    '''
        Input:
            feature_list - list of lists containing the features of objects in the considered cluster

        Output:
            mae - float: mean average errors to centroid of cluster
    '''
    def calc_mae(self, feature_list):
        centroid = np.average(feature_list, axis=0)
        mae = np.average(np.abs(feature_list - centroid[None, :]))
        return mae

    '''
        Input:
            feature_list - list of lists containing the features of objects in the considered cluster

        Output:
            max_dist - float: maximal distance of cluster member to centroid of cluster
    '''
    def calc_max_dist(self, feature_list):
        max_dist = 0
        for i in range(len(feature_list) - 1):
            for j in range(i + 1, len(feature_list)):
                cur_dist = euclidean(np.array(feature_list[i]), np.array(feature_list[j]))
                if cur_dist > max_dist:
                    max_dist = cur_dist
        max_dist /= 2 ** (1 / 2)
        return max_dist

    '''
        Input:
            feature_list - list of lists containing the features of objects in the considered cluster

        Output:
            avg_dist - float: average distance of cluster members to their minPts neighbor
    '''
    def calc_min_pts(self, feature_list):
        avg_dist = 0
        for i in range(len(feature_list)):
            dist_list = [10] * self._minPts
            for j in range(len(feature_list)):
                if i == j:
                    continue
                cur_dist = euclidean(np.array(feature_list[i]), np.array(feature_list[j]))
                for k in range(len(dist_list)):
                    if cur_dist < dist_list[k]:
                        dist_list.insert(k, cur_dist)
                        dist_list.pop(self._minPts)
            avg_dist += dist_list[self._minPts - 1]
        avg_dist /= len(feature_list)
        return avg_dist

    '''
        Output:
            rand_index - float: average rand_index for all time clusterings
    '''
    def calc_rand_rating(self):
        timestamps = self._data[self._time_column_name].unique()
        timestamps.sort()
        gr_time = self._data.groupby(self._time_column_name)
        rand_index = 0
        counter = 0
        for i in range(len(timestamps) - 1):
            for j in range(i, len(timestamps)):
                cluster_bel = gr_time[[self._object_column_name, self._cluster_column_name]].get_group(timestamps[i])
                bel_next = gr_time[[self._object_column_name, self._cluster_column_name]].get_group(timestamps[j])
                bel_next = bel_next.rename(columns={self._cluster_column_name: 'clu2'})
                cluster_bel = cluster_bel.set_index(self._object_column_name)\
                                .join(bel_next.set_index(self._object_column_name))
                true_labels = cluster_bel[self._cluster_column_name].tolist()
                pred_labels = cluster_bel['clu2'].tolist()
                rand_index += rand_score(true_labels, pred_labels)
                counter += 1
        return rand_index / counter
