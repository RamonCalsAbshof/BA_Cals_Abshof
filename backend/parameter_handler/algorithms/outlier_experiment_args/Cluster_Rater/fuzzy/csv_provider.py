from sklearn import preprocessing
import pandas as pd
import numpy as np
import os

class CSVProvider(object):

    DATA = ''
    SETTINGS = ''
    def __init__(self, settings):
        self.SETTINGS = settings

    def normalize(self, data, feature):
        f1 = data[[feature]].values.astype(float)

        min_max_scaler_dep = preprocessing.MinMaxScaler()
        f1_scaled = min_max_scaler_dep.fit_transform(f1)

        data[feature] = f1_scaled
        return data




    def get_data(self):
        if  self.SETTINGS['filename'] == 'DelayedFlights.csv':
            self.load_flights()

        else:
            self.DATA = pd.read_csv('generated_data/' + self.SETTINGS['filename'])
            self.DATA = self.DATA.rename(columns={self.SETTINGS['identifier'] : 'ObjectID', self.SETTINGS['time_column_name'] : 'Time'})
            for feature in self.SETTINGS['features']:
                self.DATA = self.DATA.rename(columns={feature : self.SETTINGS['feature_renames'][self.SETTINGS['features'].index(feature)]})

        return self.DATA