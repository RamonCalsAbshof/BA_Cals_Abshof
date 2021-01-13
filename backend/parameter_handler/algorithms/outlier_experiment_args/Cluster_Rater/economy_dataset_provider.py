import pandas as pd
import numpy as np
from sklearn import preprocessing
from sqlalchemy import create_engine
import sqlalchemy as sql


class EconomyDataset(object):

    data_table = sql.table('data')

    def __init__(self, DATASET):
        self._DATASET = DATASET

    def get_data(self):
        engine = create_engine("mysql://globaleconomy:rocrame@134.99.112.190:27020/globaleconomy2", echo=False)
        connection = engine.connect()

        command = 'SELECT `Code`, `Year`, `Month'

        for feature in self._DATASET['features']:
            command = command + '`, `' + feature

        command = command + '` FROM data WHERE Month = 12 AND Year>=' + str(self._DATASET['start_time']) + \
                  ' AND Year<=' + str(self._DATASET['end_time'])

        for feature in self._DATASET['features']:
            command = command + ' AND `' + feature + '` IS NOT NULL'

        command = command + ' ORDER BY `CODE` LIMIT 250,120'

        s = connection.execute(command)

        columns = ['ObjectID', 'Year', 'Month'] + self._DATASET['features']
        t_data = pd.DataFrame(s, columns=columns)

        # t_data = t_data.assign(Time = lambda x: x['Year'].map(str) + '-' + x['Month'].map(str) + '-01 13:00:00')
        t_data = t_data.assign(Time=lambda x: x['Year'].map(int))

        t_data = t_data[['ObjectID', 'Time'] + self._DATASET['features']]
        #
        # t_data['Time'] = pd.to_datetime(t_data['Time'])
        # t_data['Time'] = t_data['Time'].astype(np.int64) / 1000000000
        t_data = t_data.astype({'Time': int})

        for i in range(len(self._DATASET['features'])):
            t_data = t_data.rename(columns={self._DATASET['features'][i]: self._DATASET['feature_renames'][i]})
        # t_data = t_data.reset_index()
        return self.normalize(t_data)

    def normalize(self, data):
        for feature in self._DATASET['feature_renames']:
            if float(data[feature].max()) > 1.0 or float(data[feature].min()) < 0.0:
                f = data[[feature]].values.astype(float)
                min_max_scaler = preprocessing.MinMaxScaler()
                dep_scaled = min_max_scaler.fit_transform(f)
                data[feature] = dep_scaled
        return data