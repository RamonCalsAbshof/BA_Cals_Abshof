import matplotlib.pyplot as plt
from Cluster_Rater.data_plotter import DataPlotter
from Cluster_Rater.clustering.data_config import *
from Cluster_Rater.helpers import *
from Cluster_Rater.clustering.over_time_clustering import OverTimeClustering
import time

DATASET = generated_ipmu
output_path = '/home/tatusch/Dokumente/KI-Projekt/finfraud/Cluster_Rater/clustering/graph_images/'
factor_type = 'temporal_connection_sw'

plot_data = False
plot_results = True
remove_labels = False
plot_graph = False


def run_experiment():
    print('Running Over-Time Clustering on DATASET: ' + DATASET['Name'])
    data = load_data(DATASET)

    min_cf = DATASET['min_cf']
    sliding_window = DATASET['sw']

    columns = ['ObjectID', 'Time'] + DATASET['feature_renames']
    data = data[columns]

    if plot_data:
        plotter = DataPlotter(DATASET)
        plotter.plot_twod_data(data)

    start = time.perf_counter()
    otc = OverTimeClustering(data)
    start1 = time.perf_counter()

    factors = otc.get_factors(factor_type, sw=sliding_window)

    time_needed1 = time.perf_counter() - start1
    print("calculated factors in " + str(time_needed1) + " seconds.")

    start2 = time.perf_counter()

    clusters = otc.calc_clusters(factors, min_cf)

    time_needed2 = time.perf_counter() - start2
    print("calculated clusters in " + str(time_needed2) + " seconds.\n")
    time_needed = time.perf_counter() - start
    print("TOTAL CALCULATION TIME: " + str(time_needed) + " seconds.")
    clusters = otc.mark_outliers(clusters)

    if plot_results:
        data = otc.add_cluster_to_df(clusters)
        plotter = DataPlotter(DATASET)
        if len(DATASET['feature_renames']) > 1:
            plotter.plot_twodim_clusters(data, outlier=False, remove_labels=remove_labels)
        else:
            plotter.plot_onedim_clusters(data, draw_points=True)
        plt.show()

    if plot_graph:
        timestamps = data['Time'].unique()
        timestamps.sort()

        graphs = otc.create_nx_graphs(factors, min_cf)

        for i in range(len(graphs)):
            otc.save_nx_graph_fig(graphs[i],
                               output_path + DATASET['Name'] + '_' + factor_type + '_' + str(timestamps[i]) + '.png')
        plt.show()


if __name__ == "__main__":
    run_experiment()