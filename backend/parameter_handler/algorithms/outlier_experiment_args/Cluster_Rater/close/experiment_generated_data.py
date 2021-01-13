import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import itertools
import matplotlib.style as style
import os

## DBSCAN Settings
from Cluster_Rater.close.close import CLOSE

DBSCAN_METRIC = 'euclidean'
DBSCAN_METRIC_PARAMS = None
DBSCAN_ALGORITHM = 'auto'
DBSCAN_LEAF_SIZE = 30
DBSCAN_P = None
DBSCAN_N_JOBS = None

NO_TIMEPOINTS = 4
DBSCAN_MINPTS = 2
EVAL_FUNCTION = 'mse'

csv_name = os.path.join('../generated_data', 'generated_data_ipmu_trans.csv')
t_data = pd.read_csv(csv_name)


def create_dbscan_clusters(data, minpts, eps):
    clusters=pd.DataFrame()
    cluster_count = 0
    for timestamp in data['time'].unique() :
        timepoint_data = data.loc[data['time'] == timestamp]


        cluster_data = [tuple(x) for x in timepoint_data.iloc[:,2:].values]

        clustering = DBSCAN(eps, minpts, DBSCAN_METRIC,
                            DBSCAN_METRIC_PARAMS, DBSCAN_ALGORITHM,
                            DBSCAN_LEAF_SIZE, DBSCAN_N_JOBS).fit(cluster_data)



        for i in range(0,len(clustering.labels_)):
            if clustering.labels_[i] > -1:
                clustering.labels_[i] = clustering.labels_[i] + cluster_count
            else:
                clustering.labels_[i] = clustering.labels_[i] - cluster_count
        if np.max(clustering.labels_) < 0:
            cluster_count = np.max(clustering.labels_ * (-1)) + 1
        else:
            cluster_count = np.max(clustering.labels_) + 1
        idx =  np.array(range(timestamp*len(clustering.labels_)
                     ,timestamp*len(clustering.labels_)+len(clustering.labels_)))
        labels_df = pd.DataFrame({'cluster': clustering.labels_, 'idx': idx})
        labels_df = labels_df.set_index('idx')
        timepoint_data.loc[:,'idx'] = pd.Series(range(timestamp*len(clustering.labels_)
                                                    ,timestamp*len(clustering.labels_)+len(clustering.labels_)),
                                                    index=timepoint_data.index)
        timepoint_data = timepoint_data.set_index(['idx'])
        clusters = clusters.append(timepoint_data.join(labels_df))

    return clusters


def create_kmeans_clusters(data, n):
    clusters=pd.DataFrame()
    cluster_count = 0
    for timestamp in data['time'].unique() :
        timepoint_data = data.loc[data['time'] == timestamp]


        cluster_data = [tuple(x) for x in timepoint_data.iloc[:,2:].values]

        #clustering = DBSCAN(eps, minpts, DBSCAN_METRIC,
         #                   DBSCAN_METRIC_PARAMS, DBSCAN_ALGORITHM,
          #                  DBSCAN_LEAF_SIZE, DBSCAN_N_JOBS).fit(cluster_data)

        clustering = KMeans(n_clusters=n).fit(cluster_data)

        for i in range(0,len(clustering.labels_)):
            if clustering.labels_[i] > -1:
                clustering.labels_[i] = clustering.labels_[i] + cluster_count
        if np.max(clustering.labels_) < 0:
            cluster_count = np.max(clustering.labels_ * (-1)) + 1
        else:
            cluster_count = np.max(clustering.labels_) + 1
        idx =  np.array(range(timestamp*len(clustering.labels_)
                     ,timestamp*len(clustering.labels_)+len(clustering.labels_)))
        labels_df = pd.DataFrame({'cluster': clustering.labels_, 'idx': idx})
        labels_df = labels_df.set_index('idx')
        timepoint_data.loc[:,'idx'] = pd.Series(range(timestamp*len(clustering.labels_)
                                                    ,timestamp*len(clustering.labels_)+len(clustering.labels_)),
                                                    index=timepoint_data.index)
        timepoint_data = timepoint_data.set_index(['idx'])
        clusters = clusters.append(timepoint_data.join(labels_df))

    return clusters


def plotTimePoints(data, no, title='uknown parameter'):
    data.loc[data['cluster'] < 0, 'cluster'] = -1
    plot_data = data[data['time'].isin(data['time'].unique()[:no])]
    style.use('seaborn-paper')
    sns.set(font='CMU Serif', font_scale=1.8)
    # sns.set_style("darkgrid", {'axes.grid': True, 'axes.edgecolor': 'white', 'grid.color': '.9','axes.edgecolor': '.9',}, )
    g = sns.FacetGrid(plot_data, col="time", hue="cluster", col_wrap=4, palette='Set1')

    def f(x, y, z, w, **kwargs):
        ax = sns.scatterplot(x, y, **kwargs)
        ax.set(ylim=(-0.1,1.1))
        ax.set(xlim=(-0.1,1.1))
        ax.set(xlabel='feature 1', ylabel='feature 2')
        # ax.set(aspect=1.0)
        ax.set(xticks=[0.0, 0.5, 1.0])
        ax.set(yticks=[0.0, 0.5, 1.0])
        # Individual Colors
        # For every time point use 5 different colors (assume only max 4 clusters + outliers per time point)
        # Use a palette for color blind people https://davidmathlogic.com/colorblind/
        cluster_color_palette = ['#D81B60', '#1E88E5', '#FFC107', '#9C9C6E', '#1CF2AA', '#92C135', '#1033EA', '#55A1D7', '#38CA48', '#637645', '#81019B', '#E9E2A3', '#F0A054']
        cluster_colors = list()
        cluster_color_labels = list()
        for time in plot_data['time'].unique():
            #reset used colors per time point
            color_index = 1
            for cluster_id in plot_data[plot_data['time'] == time]['cluster'].unique():
                if cluster_id != -1 :
                    cluster_colors.append(cluster_color_palette[color_index])
                    cluster_color_labels.append(cluster_id)
                    color_index = color_index + 1

        cluster_colors.append(cluster_color_palette[0])
        cluster_color_labels.append(-1)

        c = lambda x: cluster_colors[cluster_color_labels.index(x)]

        for i in range(len(x)):
            ax.annotate(z.values[i], xy=(x.values[i], y.values[i]), fontsize=15,
                        xytext=(0, 0), textcoords="offset points",
                        color='black',
                        bbox=dict(boxstyle="round", alpha=0.3, color=c(w.values[i])),
                        va='center', ha='center', weight='bold')
           # print(labels[z.values[i]]+' ' + str(w.values[i]) + str(kwargs.get("color", "k")))

    g.map(f, "feature1", "feature2", "object_id", "cluster", alpha=0.6, s=5, legend='full')
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle(title)
    return plt



def normalize(data):
    Distance = data[['feature1']].values.astype(float)
    arrdelay = data[['feature2']].values.astype(float)

    min_max_scaler_dep = preprocessing.MinMaxScaler()
    dep_scaled = min_max_scaler_dep.fit_transform(Distance)

    min_max_scaler_arr = preprocessing.MinMaxScaler()
    arr_scaled = min_max_scaler_arr.fit_transform(arrdelay)

    data['feature1'] = dep_scaled
    data['feature2'] = arr_scaled

    return data


def start_KMEANS_experiment():
    max_score = -2.0
    max_k = 0.0
    min_score = 2.0
    min_k = 0.0
    mid_score = 2.0
    mid_k = 0.0

    k_list = []
    scores_list = []
    quality_list = []
    stability_list = []
    global t_data
    for k in range(1, 11, 1):
        t_data = t_data[t_data['time'].isin(t_data['time'].unique()[:NO_TIMEPOINTS])]
        c_data = normalize(t_data)
        c_data = create_kmeans_clusters(t_data, k)
        # plot = plotTimePoints(c_data, NO_TIMEPOINTS)
        columns = ['object_id', 'time', 'cluster', 'feature1', 'feature2']
        c_data = c_data[columns]
        c_data = normalize(c_data)
        rater = CLOSE(c_data)
        scores = rater.rate_clustering(return_measures=True)
        clustering_score = scores['close']
        quality_score = scores['quality']
        stability_score = scores['stability']

        if clustering_score > max_score:
            max_score = clustering_score
            print('New MAX Leader ' + str(k))
            max_k = k

        if clustering_score < min_score:
            min_score = clustering_score
            print('New MIN Leader ' + str(k))
            min_k = k

        if abs(clustering_score) < mid_score:
            mid_score = abs(clustering_score)
            print('New Mid Leader ' + str(k))
            mid_k = k
        print('k: ' + str(k))
        print('Total Clustering Score: ', str(clustering_score))
        # plot.show()

        k_list.append(k)
        scores_list.append(clustering_score)
        quality_list.append(1 - quality_score)
        stability_list.append(stability_score)

    d = {'k': k_list, 'close': scores_list, 'quality': quality_list, 'stability': stability_list}
    d_frame = pd.DataFrame(d)
    sns.set_style("whitegrid", {'axes.grid': True, 'axes.edgecolor': 'black'})
    sns.set(font='CMU Serif', font_scale=2.8, rc={'figure.figsize':(12, 6)})
    ax = sns.lineplot(x='k', y='score', hue='variable', data=pd.melt(d_frame, ['k'], value_name='score'), dashes=True)
    ax.set(xticks=np.arange(1,11,1))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], fontsize='22')
    plt.gcf().subplots_adjust(bottom=0.2)



    # print('MAX, MIN, MID ' + str(max_k) + ' ' + str(min_k) + ' ' + str(mid_k))
    dataset_name = ""
    c_data = create_kmeans_clusters(t_data, max_k)
    if csv_name == 'generated_data_pakdd_trans.csv':
        title = 'Data Set B'
        dataset_name = "B"
    elif csv_name == 'generated_data_pakdd_trans2.csv':
        title = 'Data Set A'
        dataset_name = "A"
    else:
        title = 'k='+str(max_k)
    # plt.gcf().savefig('images/' + '4_graph_trans_' + dataset_name + '.png')
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, title)
    plt.gcf().set_size_inches(10, 3)
    # plot.savefig('images/' + '4_best_trans_' + dataset_name + '.png')

    plot.show()
    plot.show()

    # c_data = create_kmeans_clusters(t_data, min_k)
    # plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'k='+str(min_k))
    # plot.show()
    #
    # c_data = create_kmeans_clusters(t_data, mid_k)
    # plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'k='+str(mid_k))
    # plot.show()
    #
    # c_data = create_kmeans_clusters(t_data, 3)
    # plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'k='+str(3))
    # plot.show()
    print(d)

def start_DBSCAN_experiment():
    max_score = -2.0
    max_epsilon = 0.0
    max_minpts = 0.0
    min_score = 2.0
    min_epsion = 0.0
    mid_score = 2.0
    mid_epsilon = 0.0

    global t_data
    t_data = t_data[t_data['time'].isin(t_data['time'].unique()[:NO_TIMEPOINTS])]


    style.use('seaborn-paper')
    sns.set(font='CMU Serif', font_scale=1.4)
    fig, ax = plt.subplots()
    for minpts in np.arange(2, 5, 1):
        eps_list = []
        scores_list = []
        stability_list = []
        mse_list = []
        minpts_list = []
        for epsilon in np.arange(0.05, 0.25, 0.01):
            c_data = t_data
            c_data = create_dbscan_clusters(c_data, minpts, epsilon)
           # plot = plotTimePoints(c_data, NO_TIMEPOINTS)
            columns = ['object_id', 'time', 'cluster', 'feature1', 'feature2']
            c_data = c_data[columns]
            c_data = normalize(c_data)
            rater = CLOSE(c_data, EVAL_FUNCTION, DBSCAN_MINPTS)
            clustering_score, stability, mse, factor = rater.rate_clustering()

            if clustering_score > max_score:
                max_score = clustering_score
                print('New MAX Leader' + str(epsilon))
                max_epsilon = epsilon
                max_minpts = minpts

            if clustering_score < 0:
                print('hallo')

            print('Epsilon: ' + str(epsilon))
            print('Total Clustering Score: ', str(clustering_score))
            #plot.show()
            eps_list.append(epsilon)
            scores_list.append(clustering_score)


        d = {'epsilon' : eps_list, 'score' : scores_list}
        d_frame = pd.DataFrame(d)

        g = sns.lineplot(x='epsilon', y='score', markers=True, label = str(minpts), alpha=0.5, data=d_frame, ax = ax)

    ax.legend().set_title('MinPts')


    #sns.legend(['minpoints = 2', 'minpoints = 3', 'minpoints = 4'], loc = 'lower right')
    sns.set(font='CMU Serif', font_scale=2)
    c_data = create_dbscan_clusters(t_data, max_minpts, max_epsilon)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS,  'epsilon ='+str(max_epsilon))
    plot.show()

    c_data = create_dbscan_clusters(t_data, DBSCAN_MINPTS, min_epsilon)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'epsilon ='+str(min_epsilon))
    plot.show()

    c_data = create_dbscan_clusters(t_data, DBSCAN_MINPTS, mid_epsilon)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'epsilon = '+str(mid_epsilon))
    plot.show()

    c_data = create_dbscan_clusters(t_data, DBSCAN_MINPTS, 0.15)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'epsilon = 0.15')
    plot.show()

    c_data = create_dbscan_clusters(t_data, DBSCAN_MINPTS, 0.25)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'epsilon = 0.25')
    plot.show()

    c_data = create_dbscan_clusters(t_data, DBSCAN_MINPTS, 0.35)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'epsilon = 0.4')
    plot.show()

    c_data = create_dbscan_clusters(t_data, DBSCAN_MINPTS, 0.55)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'epsilon = 0.55')
    plot.show()

    c_data = create_dbscan_clusters(t_data, DBSCAN_MINPTS, 0.66)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'epsilon = 0.66')
    plot.show()


def start_DBSCAN_INDIV_experiment():
    max_score = -2.0
    max_k = 0.0
    min_score = 2.0
    min_k = 0.0
    mid_score = 2.0
    mid_k = 0.0
    timepoint_clusterings = list()
    k_list = []
    scores_list = []
    current_max_cluster = -1
    global t_data
    t_data = t_data[t_data['time'].isin(t_data['time'].unique()[:NO_TIMEPOINTS])]
    t_data = normalize(t_data)
    for tp in range(0, NO_TIMEPOINTS):
        timepoint_data = t_data[t_data['time']==t_data['time'].unique()[tp]]
        timepoint_clusterings.append(list())
        for k in range(1, 6):
            c_data = create_kmeans_clusters(timepoint_data, k)
            columns = ['object_id', 'time', 'cluster', 'feature1', 'feature2']
            c_data = c_data[columns]
            timepoint_clusterings[tp].append(c_data)


    combinations = list(itertools.product(*timepoint_clusterings))
    best = 0
    for comb in combinations:
        comb[1]['cluster'] += (comb[0]['cluster'].max() + 1)
        comb[2]['cluster'] += (comb[1]['cluster'].max() + 1)
        tmp = (comb[0].append(comb[1])).append(comb[2])

        rater = CLOSE(tmp)
        clustering_score = rater.rate_clustering()
        if clustering_score > max_score:
            max_score = clustering_score
            best = tmp


    print('Best Score = ' +str(max_score))
    plot = plotTimePoints(best, NO_TIMEPOINTS, 'best_combination')
    plot.show()

# For every time point the best k is selected
def start_KMEANS_INDIV_experiment():
    max_score = -2.0
    max_k = 0.0
    min_score = 2.0
    min_k = 0.0
    mid_score = 2.0
    mid_k = 0.0
    timepoint_clusterings = list()
    k_list = []
    scores_list = []
    current_max_cluster = -1
    global t_data
    t_data = t_data[t_data['time'].isin(t_data['time'].unique()[:NO_TIMEPOINTS])]
    t_data = normalize(t_data)
    for tp in range(0, NO_TIMEPOINTS):
        timepoint_data = t_data[t_data['time']==t_data['time'].unique()[tp]]
        timepoint_clusterings.append(list())
        for k in range(1, 6):
            c_data = create_kmeans_clusters(timepoint_data, k)
            columns = ['object_id', 'time', 'cluster', 'feature1', 'feature2']
            c_data = c_data[columns]
            timepoint_clusterings[tp].append(c_data)

    combinations = list(itertools.product(*timepoint_clusterings))
    best = 0
    for comb in combinations:
        comb[1]['cluster'] += (comb[0]['cluster'].max() + 1)
        comb[2]['cluster'] += (comb[1]['cluster'].max() + 1)
        tmp = (comb[0].append(comb[1])).append(comb[2])

        rater = CLOSE(tmp)
        clustering_score = rater.rate_clustering()
        if clustering_score > max_score:
            max_score = clustering_score
            best = tmp


    print('Best Score = ' +str(max_score))
    plot = plotTimePoints(best, NO_TIMEPOINTS, 'best_combination')
    plot.show()





start_KMEANS_experiment()