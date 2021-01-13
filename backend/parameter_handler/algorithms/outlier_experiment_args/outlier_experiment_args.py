import pandas as pd
import numpy as np

import sys
import ast
import json
import os

from Cluster_Rater.mongodb_provider import MongoDB
from Cluster_Rater.csv_provider import CSVProvider
from Cluster_Rater.dbscan import DBScan
from Cluster_Rater.outlier_detection.outlier_detector import OutlierDetector
from Cluster_Rater.outlier_detection.outlier_detector_lists import OutlierDetectorLists
from Cluster_Rater.gps_dataset_provider import GPSDataset
from Cluster_Rater.outlier_detection.evaluation import *
from sqlalchemy import create_engine, MetaData, Table

restat_folder = 'label_files/'
restat_csv = 'restatements_audit_analytics.csv'

calc_eval = True
deviate = False

# Choose method: ausdm, weighted, jaccard, mldm or lists
method = 'lists'

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


def run_experiment(dataset,run_id):
    DATASET = dataset
    print('Running Outlier Detection on DATASET: ' + "financial")
    data = ''

    inserts = []

    db = MongoDB(DATASET)
    data = db.get_financial_data(True) ######

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
        for feature in DATASET['features']:
            columns.append(feature)
    else:
        columns = ['ObjectID', 'Time', 'cluster', DATASET['features'][0]]
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
            dif_restats.append((restatements_positive, restat_csv[:-4] + '_positive'))
            restatements_negative = restatements[restatements['effect'] == 'negative']
            # only consider company years with complete data for feature set
            restatements_negative = pd.merge(data, restatements_negative, on=['ObjectID', 'Time'])
            dif_restats.append((restatements_negative, restat_csv[:-4] + '_negative'))

        # in case there are restatements with positive AND negative effect
        restatements = restatements.drop_duplicates(['ObjectID', 'Time'])

        # only consider company years with complete data for feature set
        restatements = pd.merge(data, restatements, on=['ObjectID', 'Time'])
        dif_restats.append((restatements, restat_csv[:-4]))

        for labels, filename in dif_restats:
            eval_result = eval_restatements(data, result, labels)

            if deviate:
                used_data = 'dev'
            else:
                used_data = 'norm'

            inserts.append(
                    dict(
                        label = filename,
                        run_id=run_id,
                        data = used_data,
                        num_features = len(DATASET['features']),
                        precision = float(eval_result['precision']), 
                        recall = float(eval_result['recall']), 
                        f1_score = float(eval_result['f1_score']), 
                        accuracy = float(eval_result['accuracy']), 
                        true_positive = eval_result['tp'], 
                        true_negative = eval_result['tn'], 
                        false_positive = eval_result['fp'], 
                        false_negative = eval_result['fn'], 
                        avg_num_clusters = float(avg_num_clusters)
                        )
                    )
        engine = create_engine('mysql+pymysql://'\
                + os.environ['MYSQL_USER'] + ":" \
                + os.environ['MYSQL_PASSWORD'] + "@" \
                + os.environ['MYSQL_IP'] + "/" \
                + os.environ['MYSQL_DB']
                )
        conn = engine.connect()
        metadata = MetaData()
        metadata.reflect(engine)
        algorithm_table = Table(os.environ["MYSQL_DB_TABLE"],metadata,autoload=True,autoload_with=engine)
        result = conn.execute(algorithm_table.insert(), inserts)

DATASET = {"fill_missing_values":False}
DATASET.update(json.loads(sys.argv[1]))
run_id = sys.argv[2]
TAU = DATASET['tau']

run_experiment(DATASET, run_id)
 
