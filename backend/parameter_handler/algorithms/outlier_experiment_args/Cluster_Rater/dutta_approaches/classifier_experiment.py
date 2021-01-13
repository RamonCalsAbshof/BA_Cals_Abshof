import numpy as np
import pandas as pd
import itertools
import time

from mongodb_provider import MongoDB
from sqlalchemy import create_engine
from data_config import *
from sklearn.model_selection import train_test_split
from dutta_ann import *
from dutta_tree import *
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_fscore_support

np.random.seed(1) # set seed to guarantee reproducibility


resample = True
restat_csv = 'restatements_audit_analytics.csv'
balance_class_weight = False
model_name = 'perceptron'

DATASET = financial
FEATURE_SET = ['FF-CFOError', 'FF-COGSError', 'FF-INVError', 'FF-PRODError',
'FF-DISEXPError', 'FF-ACC_JONESError', 'FF-ACC_KOTHARIError',
'FF-WC_ACC0', 'FF-CH_REC', 'FF-CH_INV', 'FF-SOFT_ASSETS', 'FF-CH_CS',
'FF-CH_CM', 'FF-CH_ROA', 'FF-CH_FCF0', 'FF-TAX', 'FF-CH_EMP', 'FF-PENSION0',
'FF-CH_PENSION0','FF-ISSUE', 'FF-CFF', 'FF-RSST_ACC0']


sqldb_ann = create_engine('mysql+pymysql://root:password@134.99.112.190:27020/evaluation', pool_recycle=3600)


def insert_into_sqldb(feature_set, classifier, precision, recall, f_score, accuracy):
    table = 'classifier_evaluation'
    features = str(feature_set).strip('[]')
    sql = 'INSERT INTO ' + table + ' (feature_set, num_features, classifier, `precision`, recall, f1_score, accuracy, resampled, label_file) ' \
          'VALUES ("' + features + '",' + str(len(feature_set)) + ',"' + str(classifier) + '",' + \
          str(precision) + ',' + str(recall) + ',' + str(f_score) + ',' + str(accuracy) + ',' + str(int(resample)) + ',"' + \
          str(restat_csv) + '")'
    sqldb_ann.execute(sql)


def get_feature_dataframe(data, feature_set):
    f_data = pd.DataFrame(columns=['ObjectID', 'Time', 'Restatement'] + feature_set)
    for index, row in data[['ObjectID', 'Time', 'Restatement'] + feature_set].iterrows():
        if not np.any(pd.isna(row)):
            f_data = f_data.append(row)
    return f_data


def resample_training_data(train_X, train_Y, random_state=1):
    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(train_X, train_Y)


def run_experiment():
    start_time = time.time()

    DATASET['features'] = FEATURE_SET
    DATASET['feature_renames'] = FEATURE_SET
    db = MongoDB(DATASET)
    # get data of all companies and features at once and label them with restatements of restat_csv
    # feature values that don't exists etc are set to nan
    data = db.get_financial_train_data(csv_path=restat_csv)
    breakpoint = time.time()
    print('Data loaded in ', str(breakpoint - start_time), ' seconds.')

    for i in range(2, len(FEATURE_SET)):
        for combination in itertools.combinations(FEATURE_SET, i):
            feature_names = list(combination)
            print(str(feature_names))
            # get dataframe containing only features from considered feature set that are not nan
            f_data = get_feature_dataframe(data, feature_names)
            print('Features loaded in ', str(time.time() - breakpoint), ' seconds.\n')


            ### PREPARE DATA >> CURRENTLY ONLY ONE TIMESTAMP AT A TIME CONSIDERED (NO TIME SERIES!)

            # split dataframe into train(80%) and test(20%) set with same distribution of restatements
            train_data, test_data = train_test_split(f_data, test_size=0.2,
                                                     stratify=f_data['Restatement'], random_state=1)
            features = train_data[feature_names].values.tolist()
            labels = train_data['Restatement'].values.tolist()
            if resample:
                # resample training set with SMOTE (as in Dutta paper) to get balanced dataset
                features, labels = resample_training_data(features, labels, random_state=1)
            class_weight = None
            if balance_class_weight:
                prop_restats = np.count_nonzero(np.array(labels)) / len(labels)
                class_weight = {0: 1 - prop_restats, 1: prop_restats}
            test_features = test_data[feature_names].values.tolist()
            test_labels = test_data['Restatement'].values.tolist()
            ###

            ann = train_model(features, labels, name=model_name, class_weight=class_weight)
            ann_acc = eval_model(ann, test_features, test_labels)[1]
            ann_preds = predict_model(ann, test_features)
            ann_p, ann_r, ann_f, _ = precision_recall_fscore_support(np.around(ann_preds), test_labels)
            # print('NEURAL NETWORK')
            # print('Precision: ', str(ann_p[1]))
            # print('Recall: ', str(ann_r[1]))
            # print('F1 Score: ', str(ann_f[1]))
            # print('Accuracy: ', str(ann_acc))
            # print()
            insert_into_sqldb(feature_names, model_name, ann_p[1], ann_r[1], ann_f[1], ann_acc)
            dtree, dt_acc = run_decision_tree(features, labels, test_features, test_labels, class_weight=class_weight,
                                                print_tree=False)
            dt_preds = predict(dtree, test_features)
            dt_p, dt_r, dt_f, _ = precision_recall_fscore_support(dt_preds, test_labels)
            # print('DECISION TREE')
            # print('Precision: ', str(dt_p[1]))
            # print('Recall: ', str(dt_r[1]))
            # print('F1 Score: ', str(dt_f[1]))
            # print('Accuracy: ', str(dt_acc))
            # print()
            # print()
            insert_into_sqldb(feature_names, 'DT', dt_p[1], dt_r[1], dt_f[1], dt_acc)
            breakpoint = time.time()
    print('Needed ', str(time.time() - start_time), ' seconds.')


run_experiment()
