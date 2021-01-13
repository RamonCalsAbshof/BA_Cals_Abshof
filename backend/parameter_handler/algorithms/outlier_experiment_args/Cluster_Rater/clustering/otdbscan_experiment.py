import matplotlib.pyplot as plt
from Cluster_Rater.data_plotter import DataPlotter
from Cluster_Rater.clustering.data_config import *
from Cluster_Rater.helpers import *
from Cluster_Rater.clustering.otdbscan import OT_DBSCAN
from Cluster_Rater.dbscan import DBScan
import numpy as np

DATASET = financial

plot_data = False
plot_results = True
remove_labels = False
remove_missing_values = True
# choose between 'dbscan' and 'ot_dbscan'
method = 'ot_dbscan'

def remove_objects_with_missing_values(data):
    object_ids = set(data['ObjectID'].unique())
    timestamps = data['Time'].unique()
    for time in timestamps:
        cur_objs = data[data['Time'] == time].dropna()
        objects = set(cur_objs['ObjectID'].unique())
        object_ids = object_ids.intersection(objects)
    data = data[data['ObjectID'].isin(list(object_ids))]
    return data


def get_x_data(data):
    data = data.sort_values(['Time', 'ObjectID'])
    x_data = []
    timestamps = data['Time'].unique()
    timestamps.sort()
    for time in timestamps:
        x_data.append(data[data['Time'] == time][DATASET['feature_renames']].to_numpy())
    if len(DATASET['feature_renames']) == 1:
        x_data = np.array(x_data)[:, :, None]
    return np.array(x_data)


def add_cluster_to_df(data, clusters):
    data = data.assign(cluster=-1)
    timestamps = data['Time'].unique()
    timestamps.sort()
    object_ids = data['ObjectID'].unique()
    object_ids.sort()
    for time in range(len(timestamps)):
        for oid in range(len(object_ids)):
            data.loc[(data['Time'] == timestamps[time]) &
                           (data['ObjectID'] == object_ids[oid]), 'cluster'] = clusters[time][oid]
    return data


def run_experiment():
    print('Running TDBSCAN on DATASET: ' + DATASET['Name'])
    data = load_data(DATASET)

    columns = ['ObjectID', 'Time'] + DATASET['feature_renames']
    data = data[columns]

    if plot_data:
        plotter = DataPlotter(DATASET)
        plotter.plot_twod_data(data)

    if remove_missing_values:
        data = remove_objects_with_missing_values(data)
    x_data = get_x_data(data)

    if method == 'ot_dbscan':
        eps = DATASET['t_eps']
        min_pts = DATASET['t_minpts']
        sliding_window = DATASET['t_sw']

        if 't_eps2' in DATASET:
            eps2 = DATASET['t_eps2']
        else:
            eps2 = None

        tdbscan = OT_DBSCAN(eps=eps, min_samples=min_pts, eps2=eps2, sw=sliding_window)
        cluster_labels = tdbscan.fit_predict(x_data)
        data = add_cluster_to_df(data, cluster_labels)

    elif method == 'dbscan':
        eps = DATASET['eps']
        min_pts = DATASET['minpts']

        dbscan = DBScan(data, eps=eps, minpts=min_pts, settings=DATASET)
        data = dbscan.create_clusters()

    else:
        raise ValueError("unknown method name")

    if plot_results:
        plotter = DataPlotter(DATASET)
        if len(DATASET['feature_renames']) > 1:
            plotter.plot_twodim_clusters(data, outlier=False, remove_labels=remove_labels)
        else:
            plotter.plot_onedim_clusters(data, draw_points=True)
        plt.show()


if __name__ == "__main__":
    run_experiment()