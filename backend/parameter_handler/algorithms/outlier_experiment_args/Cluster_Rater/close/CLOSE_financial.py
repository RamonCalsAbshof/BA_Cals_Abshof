import numpy as np
import itertools

from Cluster_Rater.mongodb_provider import MongoDB
from Cluster_Rater.dbscan import DBScan
from sqlalchemy import create_engine
from Cluster_Rater.close.close import CLOSE
from Cluster_Rater.close.data_config import *


DATASET = financial
FEATURE_SET = ['FF-CFOError', 'FF-COGSError', 'FF-INVError', 'FF-PRODError',
'FF-DISEXPError', 'FF-ACC_JONESError', 'FF-ACC_KOTHARIError',
'FF-WC_ACC0', 'FF-CH_REC', 'FF-CH_INV', 'FF-SOFT_ASSETS', 'FF-CH_CS',
'FF-CH_CM', 'FF-CH_ROA', 'FF-CH_FCF0', 'FF-TAX', 'FF-CH_EMP', 'FF-PENSION0',
'FF-CH_PENSION0','FF-ISSUE', 'FF-CFF', 'FF-RSST_ACC0']

# FEATURE_SET = ['FF-CH_PENSION0','FF-ISSUE', 'FF-CFF', 'FF-RSST_ACC0']


sqldb_close = create_engine('mysql+pymysql://close:rocrame@134.99.112.190:27020/close', pool_recycle=3600)


## Settings
TAU = DATASET['tau']
MIN_PTS_START = 2
MIN_PTS_END = 10

EPSILON_START = 0.02
EPSILON_END = 0.2
EPSILON_STEP = 0.02

def insert_into_sqldb(f, close_score, min_pts, epsilon):
    table = "evaluation"
    features = str(f).strip('[]')
    sql = 'INSERT INTO '+table+' (feature_set, num_features, close_score, min_pts, epsilon) ' \
                   'VALUES ("'+features+'","'+str(len(f))+'","'+str(close_score)+'","'+ str(min_pts) +'","'+ str(epsilon) +'")'
    sqldb_close.execute(sql)


def run_experiment():

    for i in range(3, len(FEATURE_SET)):
    # for i in range(2, 3):
        for combination in itertools.combinations(FEATURE_SET, i):
            DATASET['features'] = list(combination)
            DATASET['feature_renames'] = list(combination)
            db = MongoDB(DATASET)
            data = db.get_financial_data(True)

            for minpts in range(MIN_PTS_START, MIN_PTS_END):
                for epsilon in np.arange(EPSILON_START, EPSILON_END, EPSILON_STEP):
                    dbscan = DBScan(data, minpts, epsilon, DATASET)
                    cluster_data = dbscan.create_clusters()
                    columns = ['ObjectID', 'Time', 'cluster'] + DATASET['feature_renames']
                    clusters = cluster_data[columns]

                    rater = CLOSE(clusters)
                    clustering_score = rater.rate_clustering()
                    insert_into_sqldb(DATASET['features'], clustering_score, minpts, epsilon)


run_experiment()



