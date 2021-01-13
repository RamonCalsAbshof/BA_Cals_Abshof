import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from matplotlib import pyplot as plt
import sys
import ast

from Cluster_Rater.dbscan import DBScan
from Cluster_Rater.outlier_detection.outlier_detector import OutlierDetector
from Cluster_Rater.outlier_detection.outlier_detector_lists import OutlierDetectorLists
from Cluster_Rater.data_plotter import DataPlotter
from Cluster_Rater.outlier_detection.data_config_AUSDM import *
from Cluster_Rater.outlier_detection.evaluation import *
from Cluster_Rater.helpers import *
import time

restat_folder = 'label_files/'
restat_csv = 'restatements_audit_analytics.csv'

calc_eval = True
plot_results = False
draw_points = True
deviate = False

# Choose method: ausdm, weighted, jaccard, mldm or lists
method = 'ausdm'

epsilon = range(10, 70, 5)  # will be divided by 1000
min_pts = [2, 3]
taus = range(40, 70, 5)  # will be devided by 100

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
          'VALUES (' + str(DATASET['start_year']) + ',' + str(DATASET['end_year']) + ',"' + used_data + '","' + features + '",' + \
          str(len(feature_set)) + ',' + str(precision) + ',' + str(recall) + ',' + str(f_score) + ',' + \
          str(accuracy) + ',' + str(tp) + ',' + str(tn) + ',' + str(fp) + ',' + str(fn) + ',' + str(avg_num_clusters) + \
          ',' + str(min_pts) + ',' + str(epsilon) + ',' + str(tau) + ',"' + str(label_file) + '")'
    sqldb_ann.execute(sql)


def run_experiment(dataset):
    DATASET = dataset
    # print('Running Outlier Detection on DATASET: ' + DATASET['Name'])
    data = load_data(DATASET)

    if deviate:
        data = get_dev_data(data, DATASET)
        for feature in DATASET['features']:
            data = normalize(data, feature)

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

            avg_num_clusters = num_clusters / len(time_points)

            if avg_num_clusters < 1.5:
                continue

            if len(DATASET['features']) > 1:
                columns = ['ObjectID', 'Time', 'cluster']
                for feature in DATASET['feature_renames']:
                    columns.append(feature)

            else:
                columns = ['ObjectID', 'Time', 'cluster', DATASET['feature_renames'][0]]
            clusters = cluster_data[columns]

            # start = time.perf_counter()

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

            # time_needed = time.perf_counter() - start
            # print("calculated outlier degree in " + str(time_needed) + " seconds.")

            for tau in taus:
                DATASET['tau'] = tau / 100

                # start = time.perf_counter()

                clusters, outlier_result = detector.mark_outliers(DATASET['tau'])

                # time_needed = time.perf_counter() - start
                # print("marked outlier in " + str(time_needed) + " seconds.")

                if calc_eval:
                    result = clusters[clusters['outlier'] < 0]

                    if len(result) == len(data):
                        print(avg_num_clusters)
                        continue

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

                if plot_results:
                    plotter = DataPlotter(DATASET)

                    if len(DATASET['features']) > 1:
                        plot = plotter.plot_twodim_clusters(clusters)
                        plot.show()

                    if DATASET['plot']['1DPlot']:
                        plotter.plot_onedim_clusters(clusters, outlier_result, draw_points=draw_points)

                    plt.show()



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1:] is not None:
        DATASET = dict()
        params = dict(ast.literal_eval(sys.argv[1]))
        for key in params:
            if key in ['eps', 'tau']:
                DATASET[str(key)] = float(params[key])
            elif key in ['start_year', 'end_year', 'minpts', 'no_companies']:
                DATASET[str(key)] = int(params[key])
            else:
                DATASET[str(key)] = params[key]

    else:
        DATASET = financial

    run_experiment(DATASET)
