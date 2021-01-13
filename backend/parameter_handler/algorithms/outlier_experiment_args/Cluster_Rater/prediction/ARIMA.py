import pandas as pd
import matplotlib.pyplot as plt
from Cluster_Rater.prediction.mongodb_provider import MongoDB
from Cluster_Rater.csv_provider import CSVProvider
from Cluster_Rater.prediction.data_config import *
from Cluster_Rater.dbscan import DBScan
from Cluster_Rater.data_plotter import DataPlotter
from Cluster_Rater.gps_dataset_provider import GPSDataset
from statsmodels.tsa.vector_ar.var_model import VAR
import pickle



DATASET = generated
#read the data



def run_experiment():
    print('Running Outlier Detection on DATASET: ' + DATASET['Name'])


    data = ''

    if DATASET['Name'] == 'flights':
        csv_provider = CSVProvider(DATASET)
        data = csv_provider.get_data()
        time_points = data['Time'].unique()
        time_points.sort()
        for i in range(0, len(time_points)):
            data.loc[data['Time'] == time_points[i], 'Time'] = i + 1

    elif DATASET['Name'] == 'financial':
        db = MongoDB(DATASET)
        data = db.get_financial_data(True)


    elif DATASET['Name'] == 'generated':
        csv_provider = CSVProvider(DATASET)
        data = csv_provider.get_data()

    elif DATASET['Name'] == 'gps_data':
        gps_data = GPSDataset(DATASET)
        data = gps_data.get_data()

    # Filter Data, obtain only the first no_timepoints entries
    # data = data[data['Time'].isin(data['Time'].unique()[0:DATASET['no_timepoints']])]

    if len(DATASET['features']) > 1:
        columns = ['ObjectID', 'Time', DATASET['feature_renames'][0], DATASET['feature_renames'][1]]
    else:
        columns = ['ObjectID', 'Time',  DATASET['feature_renames'][0]]
    data = data[columns]
    #data = pickle.load(open("data.pickle", "rb"))
    data.index = data.Time
    data = data.drop(['Time'], axis=1)
    print(data.dtypes)
    train_data = data[data['Time'] < DATASET['end_year']]
    test_data = data[data['Time'] == DATASET['end_year']]

    model = VAR(endog=train_data)
    model_fit = model.fit()

    # make prediction on validation
    prediction = model_fit.forecast(model_fit.y, steps=len(test_data))

    print(prediction)
    print(data.head())
    print(train_data)
    print(test_data)



run_experiment()
