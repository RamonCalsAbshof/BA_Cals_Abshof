from sklearn import preprocessing
import pandas as pd
import numpy as np

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

    def load_flights(self):
        o_data = pd.read_csv(self.SETTINGS['filename'])

        o_data['DayofMonth_prefix'] = o_data[self.SETTINGS['time_column_name']].map(str)
        temp = o_data['DayofMonth_prefix'].loc[o_data.DayofMonth < 10]
        temp = '0'+temp.map(str)
        o_data.loc[o_data.DayofMonth < 10, 'DayofMonth_prefix'] = temp

        o_data['Month_prefix'] = o_data['Month'].map(str)
        temp = o_data['Month_prefix'].loc[o_data.Month < 10]
        temp = '0'+temp.map(str)
        o_data.loc[o_data.Month < 10, 'Month_prefix'] = temp


        o_data['Datetime'] = o_data['Year'].map(str)+'-'+o_data['Month_prefix'].map(str)+'-'+o_data['DayofMonth_prefix'].map(str)
        o_data['Epoch'] = pd.to_datetime(o_data['Datetime'])
        o_data['Epoch'] = o_data['Epoch'].values.astype(np.int64) // 10 ** 6


        if len(self.SETTINGS['features']) > 1:
            t_data = pd.concat([o_data[self.SETTINGS['identifier']], o_data['Epoch'], o_data[self.SETTINGS['features'][0]], o_data[self.SETTINGS['features'][1]]], axis = 1,
                               keys = ['ObjectID', 'Time', self.SETTINGS['features'][0], self.SETTINGS['features'][1]])
        else:
            t_data = pd.concat(
                [o_data[self.SETTINGS['identifier']], o_data['Epoch'], o_data[self.SETTINGS['features'][0]]], axis=1,
                keys=['ObjectID', 'Time', self.SETTINGS['features'][0]])

        t_data = t_data.groupby(['ObjectID', 'Time']).mean().reset_index()
        
        if (self.SETTINGS['normalization'] == 'yes'):
            for feature in self.SETTINGS['features']:
                t_data = self.normalize(t_data, feature)


        time_points = t_data['Time'].unique().tolist()
        time_points.sort()
        time_points = time_points[:self.SETTINGS['no_timepoints']]

        t_data = t_data[t_data['Time'] <= time_points[len(time_points)-1]]

        self.DATA = t_data

        return t_data

    def load_diabetes_data(self):
        self.DATA = pd.read_csv(self.SETTINGS['filename'])
        self.DATA = self.DATA.rename(
            columns={self.SETTINGS['identifier']: 'ObjectID', self.SETTINGS['time_column_name']: 'Time'})
        self.DATA = self.DATA[['ObjectID', 'Time'] + self.SETTINGS['features']]
        for feature in self.SETTINGS['features']:
            self.DATA = self.DATA.rename(
                columns={feature: self.SETTINGS['features'][self.SETTINGS['features'].index(feature)]})
            self.DATA = self.normalize(self.DATA,
                                       self.SETTINGS['features'][self.SETTINGS['features'].index(feature)])
        self.DATA = self.DATA.dropna()

        object_ids = self.DATA['ObjectID'].unique()
        if 'num_objects' in self.SETTINGS and self.SETTINGS['num_objects'] > 0:
            object_ids = object_ids[:self.SETTINGS['num_objects']]
            self.DATA = self.DATA[self.DATA['ObjectID'].isin(object_ids)]

        timestamps = self.DATA['Time'].unique()
        timestamps.sort()
        if 'no_timepoints' in self.SETTINGS and self.SETTINGS['no_timepoints'] > 0:
            if '58' in self.SETTINGS['features']:
                timestamps = timestamps[1:self.SETTINGS['no_timepoints']+1]
            else:
                timestamps = timestamps[:self.SETTINGS['no_timepoints']+2]
            self.DATA = self.DATA[self.DATA['Time'].isin(timestamps)]
        return

    def get_data(self):
        if self.SETTINGS['Name'] == 'flights':
            self.load_flights()

        elif self.SETTINGS['Name'] == 'generated':
            self.DATA = pd.read_csv('generated_data/' + self.SETTINGS['filename'])
            self.DATA = self.DATA.rename(columns={self.SETTINGS['identifier'] : 'ObjectID', self.SETTINGS['time_column_name'] : 'Time'})
            for feature in self.SETTINGS['features']:
                self.DATA = self.DATA.rename(columns={feature : self.SETTINGS['features'][self.SETTINGS['features'].index(feature)]})

        else:
            self.load_diabetes_data()
        return self.DATA
