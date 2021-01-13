import numpy as np
import pandas as pd

from sqlalchemy import create_engine

from scipy import stats
import sys
import ast

from Cluster_Rater.mongodb_provider import MongoDB
from Cluster_Rater.csv_provider import CSVProvider
from Cluster_Rater.dbscan import DBScan
from Cluster_Rater.outlier_detection.outlier_detector import OutlierDetector
from Cluster_Rater.outlier_detection.reference_histogram_outlier import HistOutlier
from Cluster_Rater.data_plotter import DataPlotter
from Cluster_Rater.gps_dataset_provider import GPSDataset
from Cluster_Rater.data_config import *

restat_csv = 'label_files/restatements_audit_analytics.csv'

DATASET = ''
calc_eval = True
plot_results = False
deviate = False

epsilon = range(5, 40, 5)  # will be divided by 1000
min_pts = [2, 3]
taus = range(40, 70, 5)  # will be devided by 100

## Settings
TAU = ''

sqldb_ann = create_engine('mysql+pymysql://root:password@134.99.112.190:27020/evaluation', pool_recycle=3600)


def insert_into_sqldb(feature_set, precision, recall, f_score, accuracy, min_pts, epsilon, tau, label_file):
    table = 'outlier_ausdm_dbscan'
    features = str(feature_set).strip('[]')
    if deviate:
        used_data = 'dev'
    else:
        used_data = 'norm'

    sql = 'INSERT INTO ' + table + \
          ' (start_year, end_year, data, feature_set, num_features, `precision`, recall, f1_score, accuracy, ' \
          'min_pts, epsilon, tau, label_file) ' \
          'VALUES (' + str(DATASET['start_year']) + ',' + str(DATASET['end_year']) + ',"' + used_data + '","' + features + '",' + \
          str(len(feature_set)) + ',' + str(precision) + ',' + str(recall) + ',' + str(f_score) + ',' + \
          str(accuracy) + ',' + str(min_pts) + ',' + str(epsilon) + ',' + str(tau) + ',"' + str(label_file) + '")'
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


def calc_tau(results, percent):
    scores = results['distance'].values
    scores = np.sort(scores)
    scores = scores[::-1] # reverse order
    # if len(scores[scores == -1]) >= round(percent * len(scores)):
    #     print("More than " + str(percent) + " of subsequences are found as cluster outliers")
    scores = scores[scores != -1]
    unique_scores = np.unique(scores)
    max_outscore = scores[round(percent * len(scores)) - 1]
    while np.count_nonzero(scores >= max_outscore) > round(percent * len(scores)):
        try:
            score_index = np.where(unique_scores == max_outscore)[0][0] + 1
            max_outscore = unique_scores[score_index]
        except IndexError:
            break
    # results = results[(results['distance'] >= max_outscore) | (results['distance'] == -1)]
    results = results[results['distance'] >= max_outscore]
    return results, max_outscore


def run_experiment(dataset=DATASET):
    DATASET = dataset
    # print('Running Outlier Detection on DATASET: ' + DATASET['Name'])
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
        data = db.get_financial_data(True)  ######


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

    for eps in reversed(epsilon):
        DATASET['eps'] = eps / 1000
        for minpts in min_pts:
            DATASET['minpts'] = minpts

            # Filter Data, obtain only the first no_timepoints entries
            # data = data[data['Time'].isin(data['Time'].unique()[0:DATASET['no_timepoints']])]
            dbscan = DBScan(data, DATASET['minpts'], DATASET['eps'], DATASET)
            cluster_data = dbscan.create_clusters()
            cluster_names = cluster_data['cluster'].unique()
            num_clusters = np.count_nonzero(cluster_names >= 0)

            time_points = data['Time'].unique().tolist()
            time_points.sort()

            if num_clusters < round(1.5*len(time_points)):
                continue

            if len(DATASET['features']) > 1:
                columns = ['ObjectID', 'Time', 'cluster']
                for feature in DATASET['feature_renames']:
                    columns.append(feature)

            else:
                columns = ['ObjectID', 'Time', 'cluster', DATASET['feature_renames'][0]]
            clusters = cluster_data[columns]

            detector = OutlierDetector(clusters)
            rating = detector.calc_outlier_rating()
            # print(rating)
            outlier = HistOutlier()
            outlier_result = outlier.calc_outlier_degree(rating, clusters)

            outlier_result, DATASET['tau'] = calc_tau(outlier_result, DATASET['percent'])
            print(outlier_result)
            print(DATASET['tau'])
            # print(outlier_result)

            # mark outliers in the clusters
            clusters['outlier'] = 1
            # clusters['Time'] = clusters['Time'].astype(int)
            clusters['ObjectID'] = clusters['ObjectID'].astype(str)
            outlier_result['object_id'] = outlier_result['object_id'].astype(str)

            # mark outliers detected by distance with -1
            for index, row in outlier_result.iterrows():
                for time_point in time_points[time_points.index(int(row['start_time'])): time_points.index(
                        int(row['end_time'])) + 1]:
                    clusters.loc[(clusters['Time'] == time_point) & (
                                clusters['ObjectID'] == row['object_id']), 'outlier'] = -1

            conseq_outliers = outlier_result[outlier_result['distance'] == -1]
            # mark conseq cluster outliers with -2, mark conseq outliers which also are outliers by distance with -3
            for index, row in conseq_outliers.iterrows():
                for time_point in time_points[time_points.index(int(row['start_time'])): time_points.index(
                        int(row['end_time'])) + 1]:
                    tmp = clusters.loc[
                        (clusters['Time'] == time_point) & (clusters['ObjectID'] == row['object_id']), 'outlier']
                    if clusters.loc[(clusters['Time'] == time_point) & (
                            clusters['ObjectID'] == row['object_id']), 'outlier'].item() in [-1, -2, -3]:
                        clusters.loc[(clusters['Time'] == time_point) & (
                                    clusters['ObjectID'] == row['object_id']), 'outlier'] = -3
                    else:
                        clusters.loc[
                            (clusters['Time'] == int(time_point)) & (
                                        clusters['ObjectID'] == row['object_id']), 'outlier'] = -2

            if calc_eval:
                restatements = pd.read_csv(restat_csv, delimiter=';')
                restatements = restatements.rename(columns={'ric': 'ObjectID',
                                                            'year': 'Time'})

                restatements_positive = restatements[restatements['effect'] == 'positive']
                restatements_negative = restatements[restatements['effect'] == 'negative']

                restatements = restatements.drop_duplicates(['ObjectID', 'Time'])  # in case there are restatements with positive AND negative effect

                # only consider company years with complete data for feature set
                restatements = pd.merge(data, restatements, on=['ObjectID', 'Time'])
                restatements_positive = pd.merge(data, restatements_positive, on=['ObjectID', 'Time'])
                restatements_negative = pd.merge(data, restatements_negative, on=['ObjectID', 'Time'])

                result = clusters[clusters['outlier'] != 0]
                correct_classified = pd.merge(restatements, result, on=['ObjectID', 'Time'])

                tp = len(correct_classified)  # correct_classified
                fp = len(pd.concat(
                    [result[['ObjectID', 'Time']], correct_classified[['ObjectID', 'Time']]]).drop_duplicates(
                    subset=['ObjectID', 'Time'], keep=False))  # result - correct_classified
                fn = len(pd.concat(
                    [restatements[['ObjectID', 'Time']], correct_classified[['ObjectID', 'Time']]]).drop_duplicates(
                    subset=['ObjectID', 'Time'], keep=False))  # restatements - correct_classified
                tn = len(pd.concat([data[['ObjectID', 'Time']], result[['ObjectID', 'Time']],
                                    restatements[['ObjectID', 'Time']]]).drop_duplicates(
                    subset=['ObjectID', 'Time'], keep=False))  # all - result

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score = 2 * ((precision * recall) / (precision + recall))
                accuracy = (tp + tn) / (tp + fp + fn + tn)

                print('ALL RESTATEMENTS')
                print('Precision: ', precision)
                print('Recall: ', recall)
                print('F1-Score: ', f1_score)
                print('Accuracy: ', accuracy)

                insert_into_sqldb(DATASET['features'], precision, recall, f1_score, accuracy, DATASET['minpts'],
                                  DATASET['eps'], DATASET['tau'], restat_csv)

                correct_classified = pd.merge(restatements_positive, result, on=['ObjectID', 'Time'])
                tp = len(correct_classified.index)  # correct_classified
                fp = len(pd.concat(
                    [result[['ObjectID', 'Time']], correct_classified[['ObjectID', 'Time']]]).drop_duplicates(
                    subset=['ObjectID', 'Time'], keep=False))  # result - correct_classified
                fn = len(
                    pd.concat([restatements_positive[['ObjectID', 'Time']],
                               correct_classified[['ObjectID', 'Time']]]).drop_duplicates(
                        subset=['ObjectID', 'Time'], keep=False))  # restatements - correct_classified

                tn = len(pd.concat([data[['ObjectID', 'Time']], result[['ObjectID', 'Time']],
                                    restatements_positive[['ObjectID', 'Time']]]).drop_duplicates(
                    subset=['ObjectID', 'Time'], keep=False))  # all - result

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score = 2 * ((precision * recall) / (precision + recall))
                accuracy = (tp + tn) / (tp + fp + fn + tn)

                # print('POSITIVE EFFECT')
                # print('Recall: ', recall)

                insert_into_sqldb(DATASET['features'], precision, recall, f1_score, accuracy, DATASET['minpts'],
                                  DATASET['eps'],
                                  DATASET['tau'], restat_csv[:-4] + '_positive.csv')

                correct_classified = pd.merge(restatements_negative, result, on=['ObjectID', 'Time'])
                tp = len(correct_classified.index)  # correct_classified
                fp = len(pd.concat(
                    [result[['ObjectID', 'Time']], correct_classified[['ObjectID', 'Time']]]).drop_duplicates(
                    subset=['ObjectID', 'Time'], keep=False))  # result - correct_classified
                fn = len(
                    pd.concat([restatements_negative[['ObjectID', 'Time']],
                               correct_classified[['ObjectID', 'Time']]]).drop_duplicates(
                        subset=['ObjectID', 'Time'], keep=False))  # restatements - correct_classified

                tn = len(pd.concat([data[['ObjectID', 'Time']], result[['ObjectID', 'Time']],
                                    restatements_negative[['ObjectID', 'Time']]]).drop_duplicates(
                    subset=['ObjectID', 'Time'], keep=False))  # all - result

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score = 2 * ((precision * recall) / (precision + recall))
                accuracy = (tp + tn) / (tp + fp + fn + tn)

                # print('NEGATIVE EFFECT')
                # print('Recall: ', recall)

                insert_into_sqldb(DATASET['features'], precision, recall, f1_score, accuracy, DATASET['minpts'],
                                  DATASET['eps'],
                                  DATASET['tau'], restat_csv[:-4] + '_negative.csv')


if __name__ == "__main__":
    if sys.argv[1:] is not None:
        DATASET = dict()
        params = dict(ast.literal_eval(sys.argv[1]))
        for key in params:
            if key in ['eps', 'tau']:
                DATASET[str(key)] = float(params[key])
            elif key in ['start_year', 'end_year', 'minpts', 'no_companies']:
                DATASET[str(key)] = int(params[key])
            else:
                DATASET[str(key)] = params[key]

        TAU = float(DATASET['tau'])

    else:
        DATASET = 'financial'
        TAU = float(DATASET['tau'])

    run_experiment(DATASET)
