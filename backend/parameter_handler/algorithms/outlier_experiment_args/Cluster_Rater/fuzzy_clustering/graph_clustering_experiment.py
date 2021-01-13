import matplotlib.pyplot as plt
from Cluster_Rater.data_plotter import DataPlotter
from Cluster_Rater.Clustering.data_config import *
from Cluster_Rater.helpers import *
from Cluster_Rater.Clustering.over_time_clustering import OverTimeClustering
import time
import numpy as np

DATASET = economy
output_path = '/home/tatusch/Dokumente/KI-Projekt/finfraud/Cluster_Rater/clustering/graph_images/'
pref_type = 'temporal_sw'
plot_data = False
plot_results = True
remove_labels = False
plot_graph = True
plot_preference_graph = False
use_ranks = False


def run_experiment():
    print('Running Over-Time Clustering on DATASET: ' + DATASET['Name'])
    data = load_data(DATASET)

   # min_pref = DATASET['min_pref']
    min_pref = 0.34
    #    sliding_window = DATASET['sw']
    sliding_window = 3
    min_rank = 10

    columns = ['ObjectID', 'Time'] + DATASET['feature_renames']
    data = data[columns]

    if plot_data:
        plotter = DataPlotter(DATASET)
        plotter.plot_twod_data(data)

    start = time.perf_counter()
    otc = OverTimeClustering(data)
    start1 = time.perf_counter()

    if not use_ranks:
        preferences = otc.get_temporal_preferences_sw(sliding_window)
    else:
        preferences = otc.get_rank_temporal_preferences_sw(sliding_window)

    time_needed1 = time.perf_counter() - start1
    print("calculated preferences in " + str(time_needed1) + " seconds.")
    start2 = time.perf_counter()

    if use_ranks:
        clusters = otc.calc_clusters(preferences, min_rank, use_ranks)
    else:
        clusters = otc.calc_clusters(preferences, min_pref, use_ranks)

    time_needed2 = time.perf_counter() - start2
    print("calculated clusters in " + str(time_needed2) + " seconds.\n")
    time_needed = time.perf_counter() - start
    print("TOTAL CALCULATION TIME: " + str(time_needed) + " seconds.")
    clusters = otc.mark_outliers(clusters)

    if plot_preference_graph:
        preferences = preferences.flatten()
        preferences[::-1].sort()
        plt.plot(np.arange(len(preferences)), preferences)
        plt.show()

    if plot_results:
        data = otc.add_cluster_to_df(clusters)
        plotter = DataPlotter(DATASET)
        if len(DATASET['feature_renames']) > 1:
            plotter.plot_twodim_clusters(data, outlier=False, remove_labels=remove_labels)
        else:
            plotter.plot_onedim_clusters(data, draw_points=True)
        plt.show()

    # graphs = otc.create_preference_graphs(pref_type, min_pref)
    # timestamps = data['Time'].unique()
    # timestamps.sort()
    #
    # for i in range(len(graphs)):
    #     otc.save_graph_fig(graphs[i],
    #                        output_path + DATASET['Name'] + '_' + pref_type + '_' + str(timestamps[i]) + '.png')

    #     if plot_graph:
    #         plt.show()


if __name__ == "__main__":
    run_experiment()