import pandas as pd
import pickle
import numpy as np
import pandas as pd
import math

from sqlalchemy import create_engine

from scipy import stats
import sys
import ast

from Cluster_Rater.mongodb_provider import MongoDB
from Cluster_Rater.csv_provider import CSVProvider
from Cluster_Rater.dbscan import DBScan
from Cluster_Rater.data_plotter import DataPlotter
from Cluster_Rater.gps_dataset_provider import GPSDataset
from Cluster_Rater.prediction.data_config import *
from Cluster_Rater.close.close import *
from Cluster_Rater.prediction.close_prediction import Prediction
from Cluster_Rater.prediction.linear_regression import linear_regression


DATASET = flights
calc_eval = True
plot_results = False
deviate = False
pd.set_option('display.max_columns', 10)

## Settings
TAU = ''


sqldb_ann = create_engine('mysql+pymysql://root:password@134.99.112.190:27020/evaluation', pool_recycle=3600)


def insert_into_sqldb(feature_set, precision, recall, f_score, accuracy, min_pts, epsilon, tau, label_file):
    table = 'outlier_ausdm_dbscan'
    features = str(feature_set).strip('[]')
    sql = 'INSERT INTO ' + table + ' (feature_set, num_features, `precision`, recall, f1_score, accuracy, min_pts, epsilon, tau, label_file) ' \
          'VALUES ("' + features + '",' + str(len(feature_set)) + ',' + \
          str(precision) + ',' + str(recall) + ',' + str(f_score) + ',' + str(accuracy) + ',' + \
          str(min_pts) + ',' + str(epsilon) + ',' + str(tau) + ',"' + str(label_file) + '")'
    sqldb_ann.execute(sql)


def get_dev_data(data):
    dev_data = pd.DataFrame(columns=list(data.columns))
    rics = data['ObjectID'].values.tolist()
    for ric in rics:
        for time in range(DATASET['start_year'], DATASET['end_year']):
            cur = data[(data['ObjectID'] == ric) & (data['Time'] == time)]
            next = data[(data['ObjectID'] == ric) & (data['Time'] == time + 1)]
            if len(cur) > 0 and len(next) > 0:
                dev_dict = {'ObjectID': ric, 'Time': str(time + 1)}
                for feature in DATASET['features']:
                    dev_dict[feature] = next[feature].item() - cur[feature].item()
                dev_data = dev_data.append(dev_dict, ignore_index=True)
    return dev_data

def close(data):
    max_score = -99999999
    result = list()
    for minpts in range(DATASET['CLOSE']['minpts_start'], DATASET['CLOSE']['minpts_end']+1, DATASET['CLOSE']['minpts_step']):
        for epsilon in np.arange(DATASET['CLOSE']['eps_start'], DATASET['CLOSE']['eps_end'], DATASET['CLOSE']['eps_step']):
            dbscan = DBScan(data, minpts, epsilon, DATASET)
            cluster_data = dbscan.create_clusters()
            columns = ['ObjectID', 'Time', 'cluster'] + DATASET['feature_renames']
            clusters = cluster_data[columns]
            rater = CLOSE(clusters)
            clustering_score = rater.rate_clustering()

            if clustering_score > max_score:
                result = [clustering_score, minpts, epsilon, clusters]

    return result

''' Calculates different Errormeasures for the prediction
    Currently: Absolute Error, RMSE and Euclidean Distance
    @input data: DataFrame(columns = ['ObjectID', 'feature1',p_feature1, ..., 'featureN', 'p_featureN'])
    @output data: none
    @print output: above mentioned errors
'''


def error(data, output = True):
    #   Define error measures per row:
    data['absolute_error'] = 0
    data['euclid_distance'] = 0

    #   Calculate the absolute error
    for index, row in data.iterrows():
        for feature in DATASET['feature_renames']:
            data.loc[index, 'absolute_error']= data.loc[index, 'absolute_error'] +\
                                               math.fabs(row['p_'+feature] - row[feature])

    #   Calculate the euclidean distance
        for feature in DATASET['feature_renames']:
            data.loc[index, 'euclid_distance'] = data.loc[index, 'euclid_distance'] +\
                                                 (math.pow(row['p_'+feature] - row[feature], 2))
        data.loc[index, 'euclid_distance'] = math.sqrt(data.loc[index, 'euclid_distance'])

    if output:
        print('Average Absolute Error: ' + str(data['absolute_error'].mean()))
        print('Average Absolute Error per dimension: ' + str(data['absolute_error'].mean()/2  ))
        print('Average Euclidean Distance: ' + str(data['euclid_distance'].mean()))
        print('RMSE: ' + str(math.sqrt(data['euclid_distance'].mean())))
        #print(data)


def run_experiment():
    global DATASET
    if DATASET['Name'] == 'flights':
        csv_provider = CSVProvider(DATASET)
        data = csv_provider.get_data()
        time_points = data['Time'].unique()
        time_points.sort()
        for i in range(0, len(time_points)):
            data.loc[data['Time'] == time_points[i], 'Time'] = i + 1


    elif DATASET['Name'] == 'financial':
        db = MongoDB(DATASET)
        data = db.get_financial_data(True) ######
        #data = pickle.load(open('data.pickle', 'rb'))
        # remove companies which have no complete data (data for every year)
        for company in data['ObjectID'].unique():
            if len(data[data['ObjectID'] == company]) < (DATASET['end_year'] - DATASET['start_year']) + 1:
                data = data[data['ObjectID'] != company]
        print('There are ' + str(len(data)) + 'companies left in the data set')



    elif DATASET['Name'] == 'generated':
        csv_provider = CSVProvider(DATASET)
        data = csv_provider.get_data()


    elif DATASET['Name'] == 'gps_data':
        gps_data = GPSDataset(DATASET)
        data = gps_data.get_data()

    if deviate:
        data = get_dev_data(data)
        for feature in DATASET['features']:
            data = db.normalize(data, feature)

    squared_error = 0
    absolute_error = 0
    max_time_point = data['Time'].max()


    # Create the prediction frame which holds the expected value, the predicted value and different errors
    prediction_columns = ['ObjectID']
    for feature in DATASET['feature_renames']:
        prediction_columns.append(feature)
        prediction_columns.append('p_'+feature)
    prediction_result = pd.DataFrame(columns = prediction_columns)


    # create other models:
    lr = linear_regression(DATASET, prediction_result.copy(), data)

    # Iterate over all objects and remove the last timestamp to predict this
    for object in data[data['Time'] == max_time_point]['ObjectID'].unique():
        validation_data = data.copy()
        prediction_data = data[~((data.ObjectID == object) & (data.Time == max_time_point))].copy()


        lr.create_multidim_model(object, max_time_point, prediction_data)

    #### 2. Get the best CLOSE for the prediction set
        close_result = close(prediction_data)

    #### 3. predict the value/range for the object
        prediction = Prediction(close_result[3], object, max_time_point, DATASET)
        p = prediction.predict()


     ### 4. Calculate the Distance of the points (ERROR)
        result_dict = dict()
        result_dict['ObjectID'] = object
        for feature in DATASET['feature_renames']:
            result_dict[feature] = data[(data['ObjectID'] == object) & (data['Time'] == max_time_point)][feature].values[0]
            result_dict['p_'+feature] = p[feature]
        prediction_result = prediction_result.append(result_dict, ignore_index=True)






    error(prediction_result)
    print('Linear Regression results:')
    error(lr.get_prediction_frame())

    mixed_prediction = lr.get_prediction_frame().copy()
    for index, row in mixed_prediction.iterrows():
        for feature in DATASET['feature_renames']:
            row['p_'+feature] = row['p_'+feature] * 1/2 + prediction_result[(prediction_result['ObjectID'] == row['ObjectID'])]


run_experiment()



