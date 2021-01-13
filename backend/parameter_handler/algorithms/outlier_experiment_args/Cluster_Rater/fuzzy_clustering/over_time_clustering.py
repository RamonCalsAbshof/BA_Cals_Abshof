import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import networkx as nx
from matplotlib import pyplot as plt
import scipy.sparse.csgraph as csgraph
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import json


class OverTimeClustering(object):

    def __init__(self, data, settings):
        """
        Params:
            data (DataFrame) - pandas DataFrame with columns 'ObjectID', 'Time', features ..
                               Note: the object identifier should be in the first and the time
                                     in the second column of the DataFrame
        """
        self._data = data
        self._settings = settings

        self._column_names = data.columns.values
        self._object_column_name = self._column_names[0]
        self._time_column_name = self._column_names[1]

        self._timestamps = self._data[self._time_column_name].unique()
        self._timestamps.sort()

        self._object_ids = self._data[self._object_column_name].unique()
        self._object_ids.sort()

        self._dist_preferences = []
        self._total_preferences = []
        self._relative_preferences = []
        self._temporal_preferences = []
        self._self_preferences = []
        self._adaptability = []


    def get_preferences_df(self, pref_type):
        """
        Params:
            pref_type (str) - 'total', 'relative' or 'temporal' indicating the preference type to calculate
        Returns:
            preferences (DataFrame) - pandas DataFrame with columns 'Time', 'ObjectID1', 'ObjectID2', 'Preference'
        """
        preferences = self.get_preferences(pref_type)
        return self.create_preferences_df(preferences)

    def get_preferences(self, pref_type, **kwargs):
        """
        Params:
            pref_type (str) - 'total', 'relative' or 'temporal' indicating the preference type to calculate
        Returns:
            preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the chosen
                                preferences for all object pairs at all timestamps.
        """
        if pref_type == 'total':
            return self.get_total_preferences()
        elif pref_type == 'relative':
            return self.get_relative_preferences()
        elif pref_type == 'distance':
            return self.get_distance_preferences()
        elif pref_type == 'self':
            return self.get_self_preferences()
        elif pref_type == 'temporal':
            return self.get_temporal_preferences()
        elif pref_type == 'temporal_sw':
            return self.get_temporal_preferences_sw(**kwargs)
        else:
            print('Unknown Preference Type. Available Types are: "total", "relative", "distance", "self", '
                  '"temporal" and "temporal_sw".')
            return None

    def get_feature_vectors(self):
        """
        Returns:
            feature_vectors (array) - array with shape (num_timestamps, num_objects, num_features) containing the
                                      feature vectors for all objects at all timestamps.
                                      Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        feature_vectors = []

        for time in self._timestamps:
            feature_vectors.append(self.get_feature_vectors_at_time(time))
        return np.array(feature_vectors)

    def get_feature_vectors_at_time(self, timestamp):
        """
        Params:
            timestamp (int) - timestamp which the feature vectors should be extracted at
        Returns:
            feature_vector (array) - array with shape (num_objects, num_features) containing the feature vectors for
                                     all objects at the given timestamp.
                                     Note: The objectIDs are sorted ascending!
        """
        self._data = self._data.sort_values(by=[self._time_column_name, self._object_column_name])
        return self._data[self._data[self._time_column_name] == timestamp][self._column_names[2:]].values

    def get_total_preferences(self):
        """
        Returns:
            dist_preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the total
                                        preferences for all object pairs at all timestamps.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        if len(self._total_preferences) <= 0:
            return self.calc_total_preferences(self.get_feature_vectors())
        else:
            return self._total_preferences

    def get_relative_preferences(self):
        """
        Returns:
            relative_preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        relative preferences for all object pairs at all timestamps.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        if len(self._relative_preferences) <= 0:
            return self.calc_relative_preferences(self.get_total_preferences())
        else:
            return self._relative_preferences

    def get_distance_preferences(self):
        """
        Returns:
            relative_preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        relative preferences for all object pairs at all timestamps.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        if len(self._dist_preferences) <= 0:
            return self.calc_distance_preferences(self.get_feature_vectors())
        else:
            return self._dist_preferences

    def get_self_preferences(self):
        """
        Returns:
            relative_preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        relative preferences for all object pairs at all timestamps.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        if len(self._self_preferences) <= 0:
            return self.calc_self_preferences(self.get_distance_preferences())
        else:
            return self._self_preferences

    def get_temporal_preferences(self):
        """
        Returns:
            temporal_preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        temporal preferences for all object pairs at all timestamps.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        if len(self._temporal_preferences) <= 0:
            return self.calc_temporal_preferences(self.get_self_preferences())
        else:
            return self._temporal_preferences



    def get_temporal_preferences_sw(self, sw=3):
        """
        Returns:
            temporal_preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        temporal preferences for all object pairs at all timestamps using a sliding window.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        if len(self._temporal_preferences) <= 0:
            return self.calc_temporal_preferences_sw(self.get_self_preferences(), sw)
        else:
            return self._temporal_preferences


    def get_distance_matrix(self):
        """
        Returns:
            distance_matrix (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                      distances for all object pairs at all timestamps.
                                      Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        return self.calc_distance_matrix(self.get_feature_vectors())

    def get_clusters(self, threshold, sw=3):
        """
        Params:
            threshold (float or int)
        Optional:
            sw (int) - width of sliding window (default: 3)
        Returns:
            clusters (array) - array with shape (num_timestamps, num_objects) containing the cluster belonging of all
                               objects at all timestamps.
                               Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        return self.calc_clusters(self.get_temporal_preferences_sw(sw), threshold)



    def get_clusters_df(self, min_pref=0.5, sw=3):
        """
        Returns:
            data (DataFrame) - pandas DataFrame with columns 'ObjectID', 'Time', features.., 'cluster' containing the
                               the data and cluster belonging of all objects at all timestamps.
        """
        return self.add_cluster_to_df(self.get_clusters(min_pref, sw))

    def create_preferences_df(self, preferences):
        """
        Params:
            preferences (array-like) - containing the preference matrix with dimensions
                                       (num_timestamps, num_objects, num_objects) with preferences[i, j, k] being
                                       the preference of object j for object k at time i.
                                       Note: The timestamps as well as the objectIDs must be sorted ascending!
        Returns:
            preferences (DataFrame) - pandas DataFrame with columns 'Time', 'ObjectID1', 'ObjectID2', 'Preference'
        """
        pref = pd.DataFrame(columns=['Time', 'ObjectID1', 'ObjectID2', 'Preference'])

        for time in range(len(self._timestamps)):
            for i in range(len(self._object_ids)):
                for j in range(len(self._object_ids)):
                    pref = pref.append({'Time': self._timestamps[time],
                                        'ObjectID1': self._object_ids[i],
                                        'ObjectID2': self._object_ids[j],
                                        'Preference': preferences[time, i, j]}, ignore_index=True)
        return pref

    def add_cluster_to_df(self, clusters):
        """
        Params:
            clusters (array-like) - containing the clusters matrix with dimensions
                                       (num_timestamps, num_objects) with clusters[i, j] being
                                       the cluster belonging of object j at time i.
                                       Note: The timestamps as well as the objectIDs must be sorted ascending!
        Returns:
            data (DataFrame) - pandas DataFrame with columns 'Time', 'ObjectID', features.., 'cluster'
        """
        self._data = self._data.assign(cluster=-1)

        for time in range(len(self._timestamps)):
            for oid in range(len(self._object_ids)):
                self._data.loc[(self._data[self._time_column_name] == self._timestamps[time]) &
                               (self._data[self._object_column_name] == self._object_ids[oid]),
                               'cluster'] = clusters[time][oid]
        return self._data

    @staticmethod
    def calc_distance_matrix_at_time(feature_vectors):
        """
        Params:
            feature_vectors (array) - array with shape (num_objects, num_features) containing the feature vectors for
                                      all objects at a given timestamp
        Returns:
            distance_matrix (array) - array with shape (num_objects, num_objects) containing the distances for
                                     all object pairs at the given timestamp.
                                     Note: The objectIDs are sorted ascending!
        """
        return distance_matrix(feature_vectors, feature_vectors)

    def calc_distance_matrix(self, feature_vectors):
        """
        Params:
            feature_vectors (array) - array with shape (num_timestamps, num_objects, num_features) containing the
                                      feature vectors for all objects at all timestamps
        Returns:
            distance_matrix (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                      distances for all object pairs at all timestamps.
                                      Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        dist_matrix = []
        for time in range(len(feature_vectors)):
            dist_matrix.append(self.calc_distance_matrix_at_time(feature_vectors[time]))
        return np.array(dist_matrix)

    def calc_total_preferences_at_time(self, feature_vectors):
        """
        Params:
            feature_vectors (array) - array with shape (num_objects, num_features) containing the feature vectors at a
                                      given timestamp
        Returns:
            total_preferences (array) - array with shape (num_objects, num_objects) containing the total preferences for
                                        all object pairs at the given timestamp.
                                        Note: The objectIDs are sorted ascending!
        """
        distances = self.calc_distance_matrix_at_time(feature_vectors)
        distances = 1 - distances
        total_preferences = np.zeros(distances.shape)

        for i in range(len(distances)):
            avg = (np.sum(distances[i]) - 1) / (len(distances[i]) - 1)
            for j in range(len(distances[i])):
                if i != j:
                    total_preferences[i, j] = distances[i, j] * avg
                else:
                    total_preferences[i, j] = 1
        return total_preferences

    def calc_total_preferences(self, feature_vectors):
        """
        Params:
            feature_vectors (array) - array with shape (num_timestamps, num_objects, num_features) containing the
                                      feature vectors of all objects at all timestamps
        Returns:
            total_preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        total preferences for all object pairs at all timestamps.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        distances = self.calc_distance_preferences(feature_vectors)
        avg = (np.sum(distances, axis=1) - 1) / (len(distances[1]) - 1)

        '''
        The following Code has the same function as this commented code snipplet
        # total_preferences = []
        # for time in range(len(distances)):
        #     total_preferences.append(distances[time] * avg[time])
        # total_preferences = np.array(total_preferences)
        '''

        distances = np.moveaxis(distances, -2, 0)
        total_preferences = distances * avg
        total_preferences = np.moveaxis(total_preferences, 0, -2)
        for time in range(len(total_preferences)):
            for i in range(len(total_preferences[time])):
                total_preferences[time, i, i] = 1

        self._total_preferences = total_preferences
        return total_preferences

    @staticmethod
    def calc_relative_preferences_at_time(total_preferences):
        """
        Params:
            total_preferences (array) - array with shape (num_objects, num_objects) containing the pairwise
                                        total preferences at a given timestamp
        Returns:
            relative_preferences (array) - array with shape (num_objects, num_objects) containing the relative
                                           preferences for all object pairs at the given timestamp.
                                           Note: The objectIDs are sorted ascending!
        """
        sum_pref = np.sum(total_preferences, axis=0)
        relative_preferences = total_preferences / sum_pref
        return relative_preferences

    def calc_relative_preferences(self, total_preferences):
        """
        Params:
            total_preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        total preferences of all objects at all timestamps
        Returns:
            relative_preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        relative preferences for all object pairs at all timestamps.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        sum_pref = np.sum(total_preferences, axis=1)
        total_preferences = np.moveaxis(total_preferences, -2, 0)
        relative_preferences = total_preferences / sum_pref
        self._relative_preferences = np.moveaxis(relative_preferences, 0, -2)
        return self._relative_preferences

    def calc_distance_preferences(self, feature_vectors):
        """
        Params:
            feature_vectors (array) - array with shape (num_timestamps, num_objects, num_features) containing the
                                      feature vectors of all objects at all timestamps
        Returns:
            distance_preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        distance preferences for all object pairs at all timestamps.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        distances = self.calc_distance_matrix(feature_vectors)
        distances = (distances - np.nanmin(distances)) / (np.nanmax(distances) - np.nanmin(distances))
        distances = (1 - distances)**2
        self._dist_preferences = distances
        return self._dist_preferences



    def calc_self_preferences(self, preferences):
        """
        Params:
            preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                  preferences of all objects at all timestamps
        Returns:
            self_preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        relative preferences for all object pairs at all timestamps.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        preferences = np.moveaxis(preferences, -2, 0)
        sum_pref = (np.nansum(preferences, axis=0) - 1) / (np.count_nonzero(~np.isnan(preferences), axis=0) - 1)
        self._adaptability = sum_pref
        self_preferences = preferences * sum_pref
        self._self_preferences = np.moveaxis(self_preferences, 0, -2)
        return self._self_preferences

    def calc_temporal_preferences(self, preferences):
        """
        Params:
            preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                  preferences of all objects at all timestamps
        Returns:
            temporal_preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        temporal preferences for all object pairs at all timestamps.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        temp_preferences = np.zeros(preferences.shape)
        sum_pref = np.sum(preferences, axis=0)

        for time in range(len(preferences)):
            for i in range(len(preferences[time])):
                for j in range(len(preferences[time, i])):
                    avg_pref = (sum_pref[i, j] - preferences[time, i, j]) / \
                               (len(preferences) - 1)
                    if not np.isnan(avg_pref) and avg_pref > 0:
                        temp_preferences[time, i, j] = (preferences[time, i, j] + avg_pref) / 2
                    else:
                        temp_preferences[time, i, j] = preferences[time, i, j]
        self._temporal_preferences = temp_preferences
        return temp_preferences

    def calc_temporal_preferences_sw(self, preferences, sw=3):
        """
        Params:
            preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                  preferences of all objects at all timestamps
        Optional:
            sw (int) - width of sliding window, default:3
        Returns:
            temporal_preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        temporal preferences for all object pairs at all timestamps using a sliding window.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        temp_preferences = np.zeros(preferences.shape)

        for time in range(len(preferences)):

            prec = round((sw - 1) / 2) + 1
            if prec > time:
                prec = time

            suc = int((sw - 1) / 2)
            if suc > preferences.shape[0] - time - 1:
                suc = preferences.shape[0] - time - 1

            lsw = prec + suc

            for i in range(len(preferences[time])):
                for j in range(len(preferences[time, i])):
                    avg_pref = (np.nansum(preferences[time - prec:time + suc + 1, i, j]) - preferences[time, i, j]) / lsw
                    if not np.isnan(avg_pref) and avg_pref > 0:
                        temp_preferences[time, i, j] = (preferences[time, i, j] + avg_pref) / 2
                    else:
                        temp_preferences[time, i, j] = preferences[time, i, j]
        self._temporal_preferences = temp_preferences
        return temp_preferences

    def create_graphs(self, preferences, min_pref=0.5):
        """
        Params:
            preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the preference
                                  for all object pairs at all timestamps.
                                  Note: The timestamps as well as the objectIDs must be sorted ascending!
        Optional:
            min_pref (float) - threshold for the minimum preference for inserting edges to the graph, default:0.5
        Returns
            graphs (list of lists of tuples) - list for every point in time which holds tuples of edges.
        """
        graphs = []
        for i in range(len(self._timestamps)):
            G = list()
            for oid in range(len(self._object_ids)):
                for oid2 in range(len(self._object_ids)):
                    if preferences[i, oid, oid2] >= min_pref:
                        G.append((self._object_ids[oid], self._object_ids[oid2]))
            graphs.append(G)
        return graphs

    def create_preference_graphs(self, pref_type, min_pref=0.5, sw=3):
        """
        Params:
            pref_type (str) - 'total', 'relative' or 'temporal' indicating the preference type to calculate
        Optional:
            min_pref (float) - threshold for the minimum preference for inserting edges to the graph, default:0.5
            sw (int) - size of sliding window, only necessary if pref_type='temporal_sw', default:3
        Returns
            graphs (list) - list with length num_timestamps containing the graphs (networkx.Graph) for each timestamp
        """
        if pref_type == 'temporal_sw':
            preferences = self.get_preferences(pref_type, sw=sw)
        else:
            preferences = self.get_preferences(pref_type)

        graphs = self.create_graphs(preferences, min_pref)
        return graphs


    def get_digraph_components(self, graphs):
        """
              Params:
                  graphs (list of lists of tuples) - list for every point in time which holds tuples of edges.
              Returns:
                  components (list of components) - list of components for every timestamp
        """
        components = []
        for graph in graphs:
            G = np.zeros((len(self._object_ids),len(self._object_ids)))
            for edge in graph:
                G[np.where(self._object_ids == edge[0])[0][0],np.where(self._object_ids == edge[1])[0][0]] = 1
            G = csr_matrix(G)
            N_components, component_list = \
                connected_components(G, directed=True, return_labels=True, connection='strong')

            components.append(component_list)
        return components


    def get_component_clusters(self, components):
        """
              Params:
                  components (list of components (list of lists)) - list of components for every timestamp
              Returns:
                  clusters (df) data dataframe with appended clusters column (hard clustering)
        """
        clusters = self._data.copy()
        clusters['cluster'] = -1
        ts_conv = 0
        max_cluster = 0
        for time in clusters['Time'].unique():
            components[ts_conv] = [x+max_cluster for x in components[ts_conv]]
            clusters.loc[clusters['Time'] == time, 'cluster'] = components[ts_conv]
            for cluster in clusters.loc[clusters['Time'] == time, 'cluster'].unique():
                if len(clusters.loc[(clusters['cluster'] == cluster) & (clusters['Time'] == time)]) == 1:
                    clusters.loc[(clusters['Time'] == time) & (clusters['cluster'] == cluster), 'cluster'] = -1
            print(len(clusters[(clusters['Time'] == time) & clusters['cluster'] != -1]['cluster'].unique()))
            max_cluster = len(clusters[(clusters['Time'] == time) & clusters['cluster'] != -1]['cluster'].unique())
        return clusters


    def calc_memberships(self, clusters, max_sps):
        """
                  Params:
                      clusters: data dataframe with appended clusters column (hard clustering)
                      max_sps: dictionary of the form   {timestamp :{ Cluster1 : centroid, ..., ClusterN : centroid},
                                                         ...,
                                                         timestampM : { Cluster1 : centroid, ..., ClusterM : centroid}}
                  Returns:
                        data frame with cluster_ms  (cluster_membership) column which holds a
            """
        membership_df = clusters.copy()
        membership_df['memberships'] = str("")
        for time in clusters['Time'].unique():
            centroids = max_sps[time]
            max_preference = np.max(self._dist_preferences[np.where(self._timestamps == time)])
            for index, row in clusters[clusters['Time'] == time].iterrows():
                memberships = dict()
                for c in centroids.keys():
                    time_idx = np.where(self._timestamps == time)[0][0]
                    centroid_idx = np.where(self._object_ids == centroids[c])[0][0]
                    object_idx = np.where(self._object_ids ==  row['ObjectID'])[0][0]
                    memberships[int(c)] = self._dist_preferences[time_idx][centroid_idx][object_idx]
                membership_df.at[((clusters['Time'] == time) &
                                  (clusters['ObjectID'] == row['ObjectID'])), 'memberships'] = json.dumps(memberships)
        return membership_df

    def get_max_sp_per_cluster(self, clusters):
        """
              Params:
                  clusters (df) data dataframe with appended clusters column (hard clustering)
              Returns:

        """
        max_sp_per_cluster = dict()

        for cluster in clusters['cluster'].unique():
            if cluster != -1 and cluster != '-1':
                maximum = -999
                max_obj_id = -1

                for index, row in clusters[clusters['cluster'] == cluster].iterrows():
                    # add timestamp to dict
                    if row['Time'] not in max_sp_per_cluster:
                        max_sp_per_cluster[row['Time']] = dict()

                    # add cluster to the timestamp
                    if row['cluster'] not in max_sp_per_cluster:
                        max_sp_per_cluster[row['Time']][row['cluster']] = dict()
                    # find the object with the maximum adaptability per cluster (self preference sp)
                    # if more than one max per  cluster is wanted add it from here
                    id = np.where(self._object_ids == row['ObjectID'])[0][0]
                    time = np.where(self._timestamps == int(row['Time']))[0][0]
                    if self._adaptability[time][id] > maximum:
                        maximum = self._adaptability[time][id]
                        max_obj_id = row['ObjectID']

                #add the maximum  to the dict
                max_sp_per_cluster[row['Time']][row['cluster']] = max_obj_id


        return max_sp_per_cluster

    def show_graph(self, graphs):
        """
        Params:
            graphs (list of lists of tuples) - list for every point in time which holds tuples of edges.
        """
        ts_conv = 0
        for time in self._data['Time'].unique():
            plt.clf()
            ax = plt.axes()
            draw_data = self._data[self._data['Time'] == time].copy()
            xs = draw_data[self._settings['feature_renames'][0]]
            ys = draw_data[self._settings['feature_renames'][1]]
            draw_data.reset_index(drop=True, inplace=True)
            plt.scatter(xs, ys)

            draw_data['ObjectID'] = draw_data['ObjectID'].astype(str)

            #draw edges
            for edge in graphs[ts_conv]:
                if edge[0] != edge[1]:
                    startx = \
                        draw_data[draw_data['ObjectID']== str(edge[0])][self._settings['feature_renames'][0]].values[0]
                    starty = \
                        draw_data[draw_data['ObjectID'] == str(edge[0])][self._settings['feature_renames'][1]].values[0]
                    endx = \
                        draw_data[draw_data['ObjectID']==str(edge[1])][self._settings['feature_renames'][0]].values[0]
                    endy = \
                        draw_data[draw_data['ObjectID'] == str(edge[1])][self._settings['feature_renames'][1]].values[0]
                    ax.annotate('',
                                xy=(startx, starty),
                                xytext=(endx, endy),
                                arrowprops={'arrowstyle': '->'},
                                va='center')
            # draw labels
            cnt = 0
            for x, y in zip(xs, ys):
                plt.annotate(draw_data.at[cnt, 'ObjectID'],  # this is the text
                             (x, y),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 1),  # distance from text to points (x,y)
                             ha='center',
                             color='red')
                cnt = cnt + 1
            ts_conv = ts_conv + 1
            plt.show()

    @staticmethod
    def calc_cluster_peers_indices(preferences, min_pref):
        """
        Params:
            preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the preferences
                                  for all object pairs at all timestamps.
                                  Note: The timestamps as well as the objectIDs must be sorted ascending!
            min_pref (float) - threshold for the minimum preference for inserting edges to the graph
        Returns:
            indices (array) - array with shape (num_edges, 3) containing the lists [time, object1, object2] indicating
                              edges of the graph
        """
        return np.argwhere(preferences >= min_pref)

    @staticmethod
    def calc_cluster_peers(preferences, min_pref):
        """
        Params:
            preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the preferences
                                  for all object pairs at all timestamps.
                                  Note: The timestamps as well as the objectIDs must be sorted ascending!
            min_pref (float) - threshold for the minimum preference for inserting edges to the graph
        Returns:
            clusters (array) - array with shape (num_timestamps, num_objects, num_objets) with clusters[i, j, k] = True
                               indicating an edge between objects j and k at timestamp i.
                               Note: The timestamps as well as the objectIDs are sorted ascending!
        """

        return preferences >= min_pref


    def calc_clusters(self, preferences, threshold):
        """
        Params:
            preferences (array) - array with shape (num_timestamps, num_objects, num_objects) containing the preferences
                                  for all object pairs at all timestamps.
                                  Note: The timestamps as well as the objectIDs must be sorted ascending!
            min_pref (float) - threshold for the minimum preference for inserting edges to the graph
        Returns:
            clusters (array) - array with shape (num_timestamps, num_objects) containing the cluster belonging of all
                               objects at all timestamps.
                               Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        clusters = []
        for time in range(len(self._timestamps)):
            peers = self.calc_cluster_peers(preferences[time], threshold)
            graph = csgraph.csgraph_from_dense(peers)
            n_components, labels = csgraph.connected_components(csgraph=graph, directed=False, return_labels=True)
            clusters.append(labels)
        return clusters

    def mark_outliers(self, clusters):
        """
        Params:
            clusters (array) - array with shape (num_timestamps, num_objects) containing the cluster belongings of all
                               objects for all timestamps.
                               Note: The timestamps as well as the objectIDs must be sorted ascending!
        Returns:
            new_clusters (array) - array with shape (num_timestamps, num_objects) containing the cluster belonging of all
                               objects at all timestamps, whereby cluster with only one element are marked as noise.
                               Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        new_clusters = []
        for time in range(len(self._timestamps)):
            time_clusters = clusters[time]
            hist, _ = np.histogram(time_clusters, bins=np.arange(np.max(time_clusters)+2))
            outlier_clusters = np.argwhere(hist == 1).flatten()
            time_clusters = [x if x not in outlier_clusters else -1 for x in time_clusters ]
            new_clusters.append(np.array(time_clusters))
        return new_clusters