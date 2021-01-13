from math import sin, cos, sqrt, atan2, radians
import pandas as pd
import numpy as np
from sklearn import preprocessing
import glob
import re

class GPSDataset(object):

    SETTINGS = ''
    DATA = ''
    R = 6373.0
    FRAMES = dict()


    def __init__(self, settings):
        self.SETTINGS = settings


    def parse(self):
        li = []
        frame = pd.DataFrame()
        user_files = glob.glob(self.SETTINGS['path']+self.SETTINGS['parse_user'] + "/Trajectory/*.plt")
        for user_file in user_files:
            df = pd.read_csv(user_file, index_col=None, header=None, skiprows = 6)
            df['ObjectID'] = self.SETTINGS['parse_user']
            li.append(df)
        frame = frame.append(li)
        self.DATA = frame
        return frame

    # def parse(self):
    #     users = glob.glob('*/*')
    #     users = users[:self.SETTINGS['no_users']]
    #     li = []
    #     frame = pd.DataFrame()
    #     for user in users:
    #         user_files = glob.glob(user + "/Trajectory/*.plt")
    #         for user_file in user_files:
    #             df = pd.read_csv(user_file, index_col=None, header=None, skiprows = 6)
    #             df['ObjectID'] = user.replace('gps_data\\', '')
    #             li.append(df)
    #     frame = frame.append(li)
    #     self.DATA = frame
    #     return frame


    def datetime_to_epoch(self, o_data):
        o_data['Datetime'] = o_data[5] + ' ' + o_data[6]
        o_data['Epoch'] = pd.to_datetime(o_data['Datetime'])
        o_data['Time'] = o_data['Epoch'].values.astype(np.int64) // 10 ** 6
        return o_data

    def haversine(self, lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
        if to_radians:
            lat1 = np.radians(lat1)
            lon1 = np.radians(lon1)
            lat2 = np.radians(lat2)
            lon2 = np.radians(lon2)

        a = np.sin((lat2 - lat1) / 2.0) ** 2 + \
            np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2

        return earth_radius * 2 * np.arcsin(np.sqrt(a))

    def height_distance(selfs, start, end):
        return (start - end) * 0.3048


    def calc_distances(self, data_list):
        dist_data = []

        for frame in data_list:
            data = pd.concat([frame['ObjectID'], frame['Hour'].map(str),  frame[5].map(str) , frame[0], frame[1], frame[3]], axis = 1,
                             keys = ['ObjectID', 'Hour', 'Day', 'LONG', 'LAT', 'Height'])
            data = data.reset_index()

            data['Distance'] = self.haversine(data.LONG.shift(1), data.LAT.shift(1),
                 data.loc[1:, 'LONG'], data.loc[1:, 'LAT'])
            data['AltDistance'] = self.height_distance(data.Height.shift(1),
                                                   data.loc[1:, 'Height'])
            user_data = data.iloc[1:]
            if user_data is not None:
                dist_data.append(user_data)
        return dist_data


    def averaged_df(self, data_list):
        result = pd.DataFrame({'ObjectID': [], 'Day': [], 'Time': [],  'Distance': [], 'AltDistance':[]})
        for frame in data_list:
            object_id = str(frame['Day'].min())[8:] +'.'+ str(frame['Day'].min())[5:7]
            day = frame['Day'].min()
            time_frame = int(self.FRAMES[frame['Hour'].min()])
            distance = frame['Distance'].mean()
            #alt_distance = frame['AltDistance'].mean()
            alt_distance = 0.5
            result  = result.append({'ObjectID': object_id, 'Day': day, 'Time': time_frame, 'Distance' : distance, 'AltDistance' : alt_distance}, ignore_index=True)
        return result

    def normalize(self, data):
        Distance = data[[self.SETTINGS['feature_renames'][0]]].values.astype(float)
        arrdelay = data[[self.SETTINGS['feature_renames'][1]]].values.astype(float)

        min_max_scaler_dep = preprocessing.MinMaxScaler()
        dep_scaled = min_max_scaler_dep.fit_transform(Distance)

        min_max_scaler_arr = preprocessing.MinMaxScaler()
        arr_scaled = min_max_scaler_arr.fit_transform(arrdelay)

        data[self.SETTINGS['feature_renames'][0]] = dep_scaled
        data[self.SETTINGS['feature_renames'][1]] = arr_scaled

        return data

    def create_daily_frames(self, data):
        start = self.SETTINGS['start_time']
        end = self.SETTINGS['end_time']
        no_frames = self.SETTINGS['no_frames']
        total_hours = end - start
        frame_size = int(total_hours / no_frames)
        result = []
        for i in range(0, no_frames):
            frame_start = (start + (i) * frame_size)
            frame_end = (start + (1 + i) * frame_size)
            for hour in range(frame_start, frame_end):
                self.FRAMES[str(hour)] = str(i)

        for day in data[5].unique()[:self.SETTINGS['no_days']]:
            daily_data = data[data[5] == day]
            daily_data['Hour'] = daily_data[6].str.extract(r'\A(\d\d)', expand=True)
            daily_data['Hour'] = daily_data['Hour'].str.replace('\A0', '', regex=True)
            daily_data['Hour'] = pd.to_numeric(daily_data['Hour'])
            for i in range(0, no_frames):
                frame_start = (start + (i)*frame_size)
                frame_end = (start + (1 + i)*frame_size)
                frame_data = daily_data[daily_data['Hour'] >= frame_start]
                frame_data = frame_data[frame_data['Hour'] < frame_end]
                if frame_data.empty == False:
                    result.append(frame_data)
        return result



    def get_data(self):
        data = self.parse()
        data = self.create_daily_frames(data)
        data = self.calc_distances(data)
        data = self.averaged_df(data)
        data['Time'] = data['Time'].astype(int)
        data = data.rename(columns={self.SETTINGS['features'][0] : self.SETTINGS['feature_renames'][0], self.SETTINGS['features'][1] : self.SETTINGS['feature_renames'][1]})
        data[self.SETTINGS['feature_renames'][0]] =  data[self.SETTINGS['feature_renames'][0]] * 1000
        data.reset_index(inplace=True, drop=True)
        data = self.normalize(data)
        return data




