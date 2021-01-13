import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import sys
import ast

from Cluster_Rater.mongodb_provider import MongoDB
from Cluster_Rater.csv_provider import CSVProvider
from Cluster_Rater.dbscan import DBScan
from Cluster_Rater.outlier_detection.outlier_detector import OutlierDetector
from Cluster_Rater.outlier_detection.outlier_detector_lists import OutlierDetectorLists
from Cluster_Rater.gps_dataset_provider import GPSDataset
from Cluster_Rater.outlier_detection.evaluation import *

restat_folder = 'label_files/'
restat_csv = 'restatements_audit_analytics.csv'

calc_eval = True
deviate = False

# Choose method: ausdm, weighted, jaccard, mldm or lists
method = 'lists'


sqldb_ann = create_engine('mysql+pymysql://root:password@134.99.112.190:27020/evaluation', pool_recycle=3600)


def insert_into_sqldb(feature_set, precision, recall, f_score, accuracy, tp, tn, fp, fn, avg_num_clusters, min_pts, epsilon, tau, label_file):
    table = 'outlier_' + method + '_dbscan'

    features = str(feature_set).strip('[]')
    if deviate:
        used_data = 'dev'
    else:
        used_data = 'norm'

    sql = 'INSERT INTO ' + table + \
          ' (start_year, end_year, data, feature_set, num_features, `precision`, recall, f1_score, accuracy, ' \
          'true_positive, true_negative, false_positive, false_negative, avg_num_clusters, min_pts, epsilon, tau, label_file) ' \
          'VALUES (' + str(DATASET['start_year']) + ',' + str(
        DATASET['end_year']) + ',"' + used_data + '","' + features + '",' + \
          str(len(feature_set)) + ',' + str(precision) + ',' + str(recall) + ',' + str(f_score) + ',' + \
          str(accuracy) + ',' + str(tp) + ',' + str(tn) + ',' + str(fp) + ',' + str(fn) + ',' + str(avg_num_clusters) + \
          ',' + str(min_pts) + ',' + str(epsilon) + ',' + str(tau) + ',"' + str(label_file) + '")'
    sqldb_ann.execute(sql)


def get_dev_data(data):
    dev_data = pd.DataFrame(columns=list(data.columns))
    rics = data['ObjectID'].unique().tolist()
    for ric in rics:
        for time in range(DATASET['start_year'], DATASET['end_year']):
            cur = data[(data['ObjectID'] == ric) & (data['Time'] == time)]
            next = data[(data['ObjectID'] == ric) & (data['Time'] == time + 1)]
            if len(cur) > 0 and len(next) > 0:
                dev_dict = {'ObjectID': ric, 'Time': time + 1}
                for feature in DATASET['features']:
                    dev_dict[feature] = next[feature].item() - cur[feature].item()
                dev_data = dev_data.append(dev_dict, ignore_index=True)
    return dev_data


def run_experiment(dataset):
    DATASET = dataset
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
        data = db.get_financial_data(True) ######


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

    dbscan = DBScan(data, DATASET['minpts'], DATASET['eps'], DATASET)
    cluster_data = dbscan.create_clusters()
    cluster_names = cluster_data['cluster'].unique()
    num_clusters = np.count_nonzero(cluster_names >= 0)

    time_points = data['Time'].unique().tolist()
    time_points.sort()

    avg_num_clusters = num_clusters / len(time_points)

    if len(DATASET['features']) > 1:
        columns = ['ObjectID', 'Time', 'cluster']
        for feature in DATASET['feature_renames']:
            columns.append(feature)
    else:
        columns = ['ObjectID', 'Time', 'cluster', DATASET['feature_renames'][0]]
    clusters = cluster_data[columns]

    if method == 'ausdm':
        detector = OutlierDetector(clusters, weighting=False, jaccard=False)
    elif method == 'weighted':
        detector = OutlierDetector(clusters, weighting=True, jaccard=False)
    elif method == 'jaccard':
        detector = OutlierDetector(clusters, weighting=False, jaccard=True)
    elif method == 'mldm':
        detector = OutlierDetector(clusters, weighting=True, jaccard=True)
    elif method == 'lists':
        detector = OutlierDetectorLists(clusters)
    else:
        print('Wrong method name!')
        return

    outlier_result = detector.calc_outlier_degree()
    clusters, outlier_result = detector.mark_outliers(DATASET['tau'])

    if calc_eval:
        result = clusters[clusters['outlier'] < 0]

        restatements = pd.read_csv(restat_folder + restat_csv, delimiter=';')
        restatements = restatements.rename(columns={'ric': 'ObjectID', 'year': 'Time'})

        dif_restats = []

        if 'effect' in restatements:
            restatements_positive = restatements[restatements['effect'] == 'positive']
            # only consider company years with complete data for feature set
            restatements_positive = pd.merge(data, restatements_positive, on=['ObjectID', 'Time'])
            dif_restats.append((restatements_positive, restat_csv[:-4] + '_positive.csv'))
            restatements_negative = restatements[restatements['effect'] == 'negative']
            # only consider company years with complete data for feature set
            restatements_negative = pd.merge(data, restatements_negative, on=['ObjectID', 'Time'])
            dif_restats.append((restatements_negative, restat_csv[:-4] + '_negative.csv'))

        # in case there are restatements with positive AND negative effect
        restatements = restatements.drop_duplicates(['ObjectID', 'Time'])

        # only consider company years with complete data for feature set
        restatements = pd.merge(data, restatements, on=['ObjectID', 'Time'])
        dif_restats.append((restatements, restat_csv))

        for labels, filename in dif_restats:
            eval_result = eval_restatements(data, result, labels)
            # print(eval_result)
            insert_into_sqldb(DATASET['features'], eval_result['precision'], eval_result['recall'],
                              eval_result['f1_score'], eval_result['accuracy'], eval_result['tp'],
                              eval_result['tn'], eval_result['fp'], eval_result['fn'], avg_num_clusters,
                              DATASET['minpts'], DATASET['eps'], DATASET['tau'], filename)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DATASET = dict()
        data = dict(ast.literal_eval(sys.argv[1]))
        for key in data:
            if key in ['eps', 'tau']:
                DATASET[str(key)] = float(data[key])
            elif key in ['start_year', 'end_year', 'minpts', 'no_companies']:
                DATASET[str(key)] = int(data[key])
            else:
                DATASET[str(key)] = data[key]

        TAU = float(DATASET['tau'])

    else:
        DATASET = {'Name':'financial',
           'features':['TR-CapitalExpenditures','TR-TotalOperatingLeasesSuppl'],
           'feature_renames':['TR-CapitalExpenditures','TR-TotalOperatingLeasesSuppl'],
           'normalization_feature':'TR-TotalAssetsReported',
           'no_companies':50,
           'start_year':2008,
           'end_year':2016,
           'minpts':3,
           'eps':0.08,
           'tau':0.3
           }
        TAU = DATASET['tau']

    run_experiment(DATASET)



