import matplotlib.pyplot as plt

from Cluster_Rater.dbscan import DBScan
from Cluster_Rater.kmeans import K_Means
from Cluster_Rater.outlier_detection.outlier_detector import OutlierDetector
from Cluster_Rater.outlier_detection.outlier_detector_lists import OutlierDetectorLists
from Cluster_Rater.data_plotter import DataPlotter
from Cluster_Rater.outlier_detection.data_config_lists import *
from Cluster_Rater.helpers import *
from Cluster_Rater.clustering.over_time_clustering import OverTimeClustering as OTC
import time


DATASET = generated_oned
# Choose clustering method: dbscan, kmeans or own
cl_method = 'own'


## Settings
TAU = DATASET['tau']
# TAU = 0.3

# Choose method: ausdm, weighted, jaccard, mldm or lists
method = 'ausdm'

def remove_objects_with_missing_values(data):
    object_ids = set(data['ObjectID'].unique())
    timestamps = data['Time'].unique()
    for time in timestamps:
        objects = set(data[data['Time'] == time]['ObjectID'].unique())
        object_ids = object_ids.intersection(objects)
    data = data[data['ObjectID'].isin(list(object_ids))]
    return data

def run_experiment():
    print('Running Outlier Detection on DATASET: ' + DATASET['Name'])
    data = load_data(DATASET)
    object_column = data.columns[0]
    time_column = data.columns[1]

    if cl_method == 'kmeans':
        cluster_algorithm = K_Means(data, DATASET['k'])
    elif cl_method == 'dbscan':
        cluster_algorithm = DBScan(data, DATASET['minpts'], DATASET['eps'], DATASET)
    elif cl_method == 'own':
        data = remove_objects_with_missing_values(data)
        cluster_algorithm = OTC(data, DATASET['min_cf'], DATASET['sw'])
    else:
        print('Unknown clustering method. Choose between "dbscan", "kmeans" and "own".')
        return

    cluster_data = cluster_algorithm.create_clusters()

    columns = [object_column, time_column, 'cluster'] + DATASET['feature_renames']
    clusters = cluster_data[columns]

    start = time.perf_counter()
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
    time_needed = time.perf_counter() - start
    print("calculated outlier degree in " + str(time_needed) + " seconds.")

    start = time.perf_counter()
    clusters, outlier_result = detector.mark_outliers(DATASET['tau'])
    # clusters, outlier_result = detector.calc_statistic_cluster_outliers(factor=2)
    time_needed2 = time.perf_counter() - start
    print("marked outlier in " + str(time_needed2) + " seconds.")

    print("Total time needed for outlier detection: " + str(time_needed + time_needed2) + " seconds.")

    print(outlier_result)

    plotter = DataPlotter(DATASET)

    if len(DATASET['features']) > 1:
        plot = plotter.plot_twodim_clusters(clusters, outlier=True)
        # plot = plotter.plot_outlier_mldm(clusters)
        plot.show()

    if DATASET['plot']['1DPlot']:
        plotter.plot_onedim_clusters(clusters, outlier_result, draw_points=True)
        # plotter.plot_onedim_outlier_example(clusters, outlier_result, draw_points=True)

    plt.show()


run_experiment()