import numpy as np

class Prediction(object):

    object_id = ''
    year = ''
    data = ''
    SETTINGS = ''

    def __init__(self, d, o, y, s):
        global object_id, year, data, SETTINGS
        object_id = o
        year = y
        data = d
        SETTINGS = s

    def calc_cluster_impact(self, weights):
        # clusters to analyze:
        cluster_scores = dict()
        clusters = data[data['Time'] == year]['cluster'].unique()
        for c in clusters:
            for index, row in data[data['cluster'] == c].iterrows():
                try:
                    if str(c) not in cluster_scores.keys():
                        cluster_scores[str(c)] = {'score': 0, 'count': 0}
                    cluster_scores[str(c)]['score'] = cluster_scores[str(c)]['score'] + weights[str(row['ObjectID'])]
                    cluster_scores[str(c)]['count'] = cluster_scores[str(c)]['count'] + 1
                except KeyError:
                    print(row['ObjectID'] + 'has no score!')
        return(cluster_scores)


    def calculate_weights(self):
        global object_id, year, data, SETTINGS
        # Set all object weights to 0
        weighted_data = data.copy()
        weighted_data['weight'] = 0
        min_time = data['Time'].min()

        # Set weights for time points which share clusters with the TS to predict
        for time_point in weighted_data['Time'].unique():
            # just look at previous years
            if time_point != year:
                object_weight = 1 * (time_point/(year-min_time))
                #object_weight = 1/(year - time_point)
                current_object_cluster = weighted_data[(weighted_data['Time'] == time_point) & (weighted_data['ObjectID'] == object_id)]['cluster'].values[0]
                if current_object_cluster != -1:
                    weighted_data.loc[weighted_data['cluster'] == current_object_cluster, "weight"] = object_weight
                else:
                    weighted_data.loc[weighted_data['cluster'] == current_object_cluster, "weight"] = 0

        # Calculate the TS weights:
        ts_weights = dict()
        ts_weights_total = 0
        for ts in weighted_data['ObjectID'].unique():
            ts_weights[str(ts)] = weighted_data[weighted_data['ObjectID'] == ts]['weight'].sum()
            ts_weights_total = ts_weights_total + ts_weights[str(ts)]

        # Normalize ts weights:
        for ts in ts_weights.keys():
            ts_weights[str(ts)] = ts_weights[str(ts)] / ts_weights_total


        return ts_weights

    def predict(self):
        global object_id, year, data, SETTINGS
        weights = self.calculate_weights()
        prediction = dict()
        for feature in SETTINGS['feature_renames']:
            prediction[feature] = 0
        regarded_objects = 0
        for index, row in data[data['Time'] == year].iterrows():
            try:
                for feature in SETTINGS['feature_renames']:
                    object_id = row['ObjectID']
                    if type(object_id) == np.float64:
                        object_id = str(int(object_id))
                    else:
                        object_id = str(object_id)
                    prediction[feature] = prediction[feature] + weights[object_id] * row[feature]
                regarded_objects = regarded_objects + 1
            except KeyError:
                print('Object ' + object_id + ' is not available')
        for feature in SETTINGS['feature_renames']:
            prediction[feature] = prediction[feature]

        return prediction









