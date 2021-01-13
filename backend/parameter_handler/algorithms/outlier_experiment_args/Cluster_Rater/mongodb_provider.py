from pymongo import MongoClient
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os

class MongoDB(object):

    db = ''
    SETTINGS = ''
    def __init__(self, settings):
        client = MongoClient('134.99.112.190',
                             username='read_user',
                             password='tepco11x?z',
                             authSource='finfraud3',
                             authMechanism='SCRAM-SHA-1',
                             port=27017)
        self.db = client.finfraud3.original
        self.SETTINGS = settings

    def normalize(self, data, feature):
        f1 = data[[feature]].values.astype(float)

        min_max_scaler_dep = preprocessing.MinMaxScaler()
        f1_scaled = min_max_scaler_dep.fit_transform(f1)

        data[feature] = f1_scaled
        return data


    def get_financial_data(self, nd=False):
        if nd == True:
            return self.get_financial_data_nd()
        if len(self.SETTINGS['features']) < 2:
            return self.get_financial_data_oned()
        else:
            return self.get_financial_data_twod()






    def get_financial_data_twod(self):
        no_of_companies = self.SETTINGS['no_companies']
        normalization_feature = self.SETTINGS['normalization_feature']
        features = self.SETTINGS['features']
        feature_rename = self.SETTINGS['features']
        i = 0
        series_list = list()
        for document in self.db.find():
            if no_of_companies != 0:
                if i == no_of_companies:
                    break
                else:
                    i = i + 1
            if document['ric'] != 'nan':
                for year in range(self.SETTINGS['start_year'], self.SETTINGS['end_year']+1, 1):
                    try:
                        if ((document[str(year)][features[0]] != "nan") and (
                                document[str(year)][features[0]] != '') and (
                                document[str(year)][features[0]] != '0')
                                and (document[str(year)][features[1]] != "nan") and (
                                        document[str(year)][features[1]] != '') and (
                                        document[str(year)][features[1]] != '0')
                                and (document[str(year)][normalization_feature] != "nan") and (
                                        document[str(year)][normalization_feature] != '') and (
                                        document[str(year)][normalization_feature] != '0')):
                            ric = document['ric']
                            feature_1 = float(document[str(year)][features[0]]) / float(
                                document[str(year)][normalization_feature])
                            feature_2 = float(document[str(year)][features[1]]) / float(
                                document[str(year)][normalization_feature])
                            series_list.append(pd.Series([ric, str(year), feature_1, feature_2],
                                                         index=['ObjectID', 'Time', feature_rename[0], feature_rename[1]]))
                    except ZeroDivisionError:
                        print('divided by zero')
        t_data = pd.DataFrame(series_list, columns=['ObjectID', 'Time', feature_rename[0], feature_rename[1]])
        t_data['Time'] = pd.to_numeric(t_data['Time'])

        tmp_data = t_data.groupby(['ObjectID', 'Time']).mean().reset_index()
        del t_data
        t_data = tmp_data
        t_data = self.normalize(t_data, feature_rename[0])
        t_data = self.normalize(t_data, feature_rename[1])
        return t_data

    def get_financial_data_oned(self):
        no_of_companies = self.SETTINGS['no_companies']
        normalization_feature = self.SETTINGS['normalization_feature']
        features = self.SETTINGS['features']
        feature_rename = self.SETTINGS['features']
        i = 0
        series_list = list()
        for document in self.db.find():
            if i == no_of_companies:
                break
            else:
                i = i + 1
            if document['ric'] != 'nan':
                for year in range(self.SETTINGS['start_year'], self.SETTINGS['end_year'] + 1, 1):
                    try:
                        if ((document[str(year)][features[0]] != "nan") and (
                                document[str(year)][features[0]] != '') and (
                                document[str(year)][features[0]] != '0')
                                and (document[str(year)][normalization_feature] != "nan") and (
                                        document[str(year)][normalization_feature] != '') and (
                                        document[str(year)][normalization_feature] != '0')):
                            ric = document['ric']
                            feature_1 = float(document[str(year)][features[0]]) / float(
                                document[str(year)][normalization_feature])
                            series_list.append(pd.Series([ric, str(year), feature_1, ],
                                                         index=['ObjectID', 'Time', feature_rename[0]
                                                                ]))
                    except ZeroDivisionError:
                        print('divided by zero')
        t_data = pd.DataFrame(series_list, columns=['ObjectID', 'Time', feature_rename[0]])
        t_data['Time'] = pd.to_numeric(t_data['Time'])

        tmp_data = t_data.groupby(['ObjectID', 'Time']).mean().reset_index()
        del t_data
        t_data = tmp_data
        t_data = self.normalize(t_data, feature_rename[0])
        return t_data


    def get_financial_data_nd(self):
        no_of_companies = self.SETTINGS['no_companies']
        normalization_feature = self.SETTINGS['normalization_feature']
        features = self.SETTINGS['features']
        feature_rename = self.SETTINGS['features']
        i = 0
        series_list = list()
        for document in self.db.find():
            if no_of_companies != 0:
                if i == no_of_companies:
                    break
                else:
                    i = i + 1
            if document['ric'] != 'nan':
                ric = document['ric']
                for year in range(self.SETTINGS['start_year'], self.SETTINGS['end_year'] + 1, 1):
                    all_features_available = True
                    feature_set = list()
                    for feature in features:
                        try:
                            if ((document[str(year)][feature] == "nan") or (
                                    document[str(year)][feature] == '') or (
                                    document[str(year)][feature] == '0')):
                                feature_set.append(False)
                            elif ((normalization_feature!='') and ((document[str(year)][normalization_feature] == "nan") or (
                                    document[str(year)][normalization_feature] == '') or (
                                    document[str(year)][normalization_feature] == '0'))):
                                feature_set.append(False)
                            else:
                                if normalization_feature != '':
                                    if "TR" in feature:
                                        try:
                                            feature_set.append(float(document[str(year)][feature])/
                                                               float(document[str(year)][normalization_feature]))
                                        except ZeroDivisionError:
                                            feature_set.append(False)
                                    else:
                                        feature_set.append(float(document[str(year)][feature]))
                                else:
                                    feature_set.append(float(document[str(year)][feature]))
                        except KeyError:
                            feature_set.append(False)

                    if False not in feature_set:
                        time_data = [ric, str(year)] + feature_set
                        index_data = ['ObjectID', 'Time'] + feature_rename
                        series_list.append(pd.Series(time_data,
                                                     index=index_data))

        index_data = ['ObjectID', 'Time'] + feature_rename
        t_data = pd.DataFrame(series_list, columns=index_data)
        t_data['Time'] = pd.to_numeric(t_data['Time'])

        tmp_data = t_data.groupby(['ObjectID', 'Time']).mean().reset_index()
        del t_data
        t_data = tmp_data
        for feature in feature_rename:
            t_data = self.normalize(t_data, feature)


        # If we need nan values for missing data:
        if self.SETTINGS['fill_missing_values']:
            object_ids = t_data['ObjectID'].unique()
            time_points = t_data['Time'].unique()
            for object in object_ids:
                for year in time_points:
                    check = t_data[(t_data['ObjectID']==object) & (t_data['Time'] == year)]

                    if len(check) == 0:
                        dummy_row = dict({'ObjectID': object, 'Time': year})
                        for feature in self.SETTINGS['features']:
                            dummy_row[feature] = np.nan
                        t_data = t_data.append(dummy_row, ignore_index=True)


        return t_data

    def get_financial_train_data(self, csv_path='restatements_all.csv'):
        no_of_companies = self.SETTINGS['no_companies']
        normalization_feature = self.SETTINGS['normalization_feature']
        features = self.SETTINGS['features']
        feature_rename = self.SETTINGS['features']

        restatements = pd.read_csv(csv_path, delimiter=';')

        i = 0
        series_list = list()
        for document in self.db.find():
            if no_of_companies != 0:
                if i == no_of_companies:
                    break
                else:
                    i = i + 1
            if document['ric'] != 'nan':
                ric = document['ric']
                for year in range(self.SETTINGS['start_year'], self.SETTINGS['end_year'] + 1, 1):
                    all_features_available = True
                    feature_set = list()
                    for feature in features:
                        try:
                            if feature.startswith('TR'):
                                if ((document[str(year)][features[0]] != "nan") and (
                                        document[str(year)][features[0]] != '') and (
                                        document[str(year)][features[0]] != '0')
                                        and (document[str(year)][normalization_feature] != "nan") and (
                                                document[str(year)][normalization_feature] != '') and (
                                                document[str(year)][normalization_feature] != '0')):
                                    feature_set.append(document[str(year)][feature] / document[str(year)][normalization_feature])
                                else:
                                    feature_set.append(None)
                            elif ((document[str(year)][feature] == "nan") or (
                                    document[str(year)][feature] == '') or (
                                    document[str(year)][feature] == '0')):
                                feature_set.append(None)
                            else:
                                feature_set.append(document[str(year)][feature])
                        except KeyError:
                            feature_set.append(None)

                    if len(restatements[(restatements['ric'] == ric) & (restatements['year'] == year)]) > 0:
                        restated = 1
                    else:
                        restated = 0
                    time_data = [ric, str(year), restated] + feature_set
                    index_data = ['ObjectID', 'Time', 'Restatement'] + feature_rename
                    series_list.append(pd.Series(time_data,
                                                 index=index_data))

        index_data = ['ObjectID', 'Time', 'Restatement'] + feature_rename
        t_data = pd.DataFrame(series_list, columns=index_data)
        t_data['Time'] = pd.to_numeric(t_data['Time'])

        tmp_data = t_data.groupby(['ObjectID', 'Time']).mean().reset_index()
        del t_data
        t_data = tmp_data
        for feature in feature_rename:
            t_data = self.normalize(t_data, feature)
        return t_data
