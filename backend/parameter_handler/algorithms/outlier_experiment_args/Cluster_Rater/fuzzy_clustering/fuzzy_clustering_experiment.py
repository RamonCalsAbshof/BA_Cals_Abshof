from Cluster_Rater.data_plotter import DataPlotter
from Cluster_Rater.fuzzy_clustering.data_config import *
from Cluster_Rater.helpers import *
from Cluster_Rater.fuzzy_clustering.over_time_clustering import OverTimeClustering

DATASET = generated
output_path = '/home/tatusch/Dokumente/KI-Projekt/finfraud/Cluster_Rater/clustering/graph_images/'
pref_type = 'temporal_sw'
plot_results = True
remove_labels = False
plot_graph = True
plot_preference_graph = False



def run_experiment():
    print('Running Over-Time Clustering on DATASET: ' + DATASET['Name'])
    data = load_data(DATASET)

   # min_pref = DATASET['min_pref']
    min_pref = 0.32
    #    sliding_window = DATASET['sw']
    sliding_window = 3
    min_rank = 10

    columns = ['ObjectID', 'Time'] + DATASET['feature_renames']
    data = data[columns]

    otc = OverTimeClustering(data, DATASET)


    preferences = otc.get_temporal_preferences_sw(sliding_window)
    #graphs = otc.create_preferences_df(preferences)
    graphs = otc.create_graphs(preferences, min_pref)
    comp = otc.get_digraph_components(graphs)
    comp_clusters = otc.get_component_clusters(comp)
    centroids = otc.get_max_sp_per_cluster(comp_clusters)
    memberships = otc.calc_memberships(comp_clusters, centroids)



    if plot_results:
        plotter = DataPlotter(DATASET)
        plt = plotter.plot_twodim_fuzzy(memberships, outlier=False, remove_labels=remove_labels)
        plt.show()

   # otc.show_graph(graphs)


if __name__ == "__main__":
    run_experiment()