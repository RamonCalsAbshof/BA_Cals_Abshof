import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import itertools
import matplotlib.style as style

## DBSCAN Settings
from Cluster_Rater.close.close import CLOSE

DBSCAN_METRIC = 'euclidean'
DBSCAN_METRIC_PARAMS = None
DBSCAN_ALGORITHM = 'auto'
DBSCAN_LEAF_SIZE = 30
DBSCAN_P = None
DBSCAN_N_JOBS = None

NO_TIMEPOINTS = 3
DBSCAN_MINPTS = 2
EVAL_FUNCTION = 'mae'

o_data = pd.read_csv('DelayedFlights.csv')

o_data['DayofMonth_prefix'] = o_data['DayofMonth'].map(str)
temp = o_data['DayofMonth_prefix'].loc[o_data.DayofMonth < 10]
temp = '0'+temp.map(str)
o_data['DayofMonth_prefix'].loc[o_data.DayofMonth < 10] = temp

o_data['Month_prefix'] = o_data['Month'].map(str)
temp = o_data['Month_prefix'].loc[o_data.Month < 10]
temp = '0'+temp.map(str)
o_data['Month_prefix'].loc[o_data.Month < 10] = temp


o_data['Datetime'] = o_data['Year'].map(str)+'-'+o_data['Month_prefix'].map(str)+'-'+o_data['DayofMonth_prefix'].map(str)
o_data['Epoch'] = pd.to_datetime(o_data['Datetime'])
o_data['Epoch'] = o_data['Epoch'].values.astype(np.int64) // 10 ** 6



t_data = pd.concat([o_data['UniqueCarrier'], o_data['Epoch'], o_data['Distance'], o_data['ArrDelay']], axis = 1,
                   keys = ['ObjectID', 'Time', 'Distance', 'ArrDelay'])

t_data = t_data.groupby(['ObjectID', 'Time']).mean().reset_index()


def create_dbscan_clusters(data, minpts, eps):
    clusters=pd.DataFrame()
    cluster_count = 0
    for timestamp in data['Time'].unique() :
        timepoint_data = data.loc[data['Time'] == timestamp]


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
    for timestamp in data['Time'].unique() :
        timepoint_data = data.loc[data['Time'] == timestamp]


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
    plot_data = data[data['Time'].isin(data['Time'].unique()[:no])]
    plot_data.rename(columns={'Time': 'Day'}, inplace=True)
    plot_data.loc[plot_data['Day'] == 1206057600000, 'Day'] = 80
    plot_data.loc[plot_data['Day'] == 1206144000000, 'Day'] = 81
    plot_data.loc[plot_data['Day'] == 1206230400000, 'Day'] = 82
    sns.set(font='CMU Serif')
    style.use('seaborn-paper')
    sns.set(font='CMU Serif', font_scale=1.4)
    g = sns.FacetGrid(plot_data, col="Day", hue="cluster", col_wrap = 3, palette='Set1')
    def f(x, y, z, w, **kwargs):
        labels = {'9E': 'A',
                  'AA': 'B',
                  'AQ': 'C',
                  'AS': 'D',
                  'B6': 'E',
                  'CO': 'F',
                  'DL': 'G',
                  'EV': 'H',
                  'F9': 'I',
                  'FL': 'J',
                  'HA': 'K',
                  'MQ': 'L',
                  'NW': 'M',
                  'OH': 'N',
                  'OO': 'O',
                  'UA': 'P',
                  'US': 'Q',
                  'WN': 'R',
                  'XE': 'S',
                  'YV': 'T',}
        ax = sns.scatterplot(x, y, **kwargs)
        ax.set(ylim=(-0.2,1.1))
        ax.set(xlim=(-0.2,1.1))
        # Individual Colors
        # For every time point use 5 different colors (assume only max 4 clusters + outliers per time point)
        # Use a palette for color blind people https://davidmathlogic.com/colorblind/
        cluster_color_palette = ['#D81B60', '#1E88E5', '#FFC107', '#9C9C6E', '#1CF2AA', '#92C135', '#1033EA', '#55A1D7',
                                 '#38CA48', '#637645', '#81019B', '#E9E2A3', '#F0A054']
        cluster_colors = list()
        cluster_color_labels = list()
        for time in plot_data['Day'].unique():
            # reset used colors per time point
            color_index = 1
            for cluster_id in plot_data[plot_data['Day'] == time]['cluster'].unique():
                if cluster_id != -1:
                    cluster_colors.append(cluster_color_palette[color_index])
                    cluster_color_labels.append(cluster_id)
                    color_index = color_index + 1

        cluster_colors.append(cluster_color_palette[0])
        cluster_color_labels.append(-1)

        c = lambda x: cluster_colors[cluster_color_labels.index(x)]

        for i in range(len(x)):
            ax.annotate(z.values[i], xy=(x.values[i], y.values[i]), fontsize=12,
                        xytext=(0, 0), textcoords="offset points",
                        color='black',
                        bbox=dict(boxstyle="round", alpha=0.3, color=c(w.values[i])),
                        va='center', ha='center', weight='bold', alpha=1)
        # print(labels[z.values[i]]+' ' + str(w.values[i]) + str(kwargs.get("color", "k")))

    g.map(f, "Distance", "ArrDelay", "ObjectID", "cluster", alpha=0.6, s=5, legend='full')
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle(title)
    return plt


def normalize(data):
    Distance = data[['Distance']].values.astype(float)
    arrdelay = data[['ArrDelay']].values.astype(float)

    min_max_scaler_dep = preprocessing.MinMaxScaler()
    dep_scaled = min_max_scaler_dep.fit_transform(Distance)

    min_max_scaler_arr = preprocessing.MinMaxScaler()
    arr_scaled = min_max_scaler_arr.fit_transform(arrdelay)

    data['Distance'] = dep_scaled
    data['ArrDelay'] = arr_scaled

    return data

def start_DBSCAN_experiment():
    max_score = -2.0
    max_epsilon = 0.0
    max_minpts = 0.0
    min_score = 2.0
    min_epsion = 0.0
    mid_score = 2.0
    mid_epsilon = 0.0

    global t_data
    t_data = t_data[t_data['Time'].isin(t_data['Time'].unique()[80:NO_TIMEPOINTS + 80])]
    style.use('seaborn-paper')
    sns.set(font='CMU Serif', font_scale=1.4)
    fig, ax = plt.subplots()

    eps_list = []
    scores_list = []
    stability_list = []
    mse_list = []
    minpts_list = []
    factor_list = []
    for epsilon in np.arange(0.05, 0.25, 0.01):
        c_data = normalize(t_data)
        c_data = create_dbscan_clusters(c_data, DBSCAN_MINPTS, epsilon)
       # plot = plotTimePoints(c_data, NO_TIMEPOINTS)
        columns = ['ObjectID', 'Time', 'cluster', 'Distance', 'ArrDelay']
        c_data = c_data[columns]
        c_data = normalize(c_data)
        rater = CLOSE(c_data, EVAL_FUNCTION, DBSCAN_MINPTS)
        clustering_score, stability, mse, factor = rater.rate_clustering()

        if clustering_score > max_score:
            max_score = clustering_score
            print('New MAX Leader' + str(epsilon))
            max_epsilon = epsilon


        if clustering_score < 0:
            print('hallo')

        print('Epsilon: ' + str(epsilon))
        print('Total Clustering Score: ', str(clustering_score))
        #plot.show()
        eps_list.append(epsilon)
        scores_list.append(clustering_score)
        mse_list.append((1-mse))
        factor_list.append(factor)
        stability_list.append(stability)


    d = {'epsilon' : eps_list, 'score' : scores_list}
    d_frame = pd.DataFrame(d)

    sns.lineplot(x='epsilon', y='score', markers=True, label = 'score', data=d_frame, ax = ax)

    d = {'epsilon': eps_list, 'mse': mse_list}
    d_frame = pd.DataFrame(d)
    sns.lineplot(x='epsilon', y='mse', markers=True, label='1 - mse', data=d_frame, ax=ax)

    d = {'epsilon': eps_list, 'stability': stability_list}
    d_frame = pd.DataFrame(d)
    sns.lineplot(x='epsilon', y='stability', markers=True, label='stability', data=d_frame, ax=ax)

    d = {'epsilon': eps_list, 'factor': factor_list}
    d_frame = pd.DataFrame(d)
    sns.lineplot(x='epsilon', y='factor', markers=True, label='pre-factor', data=d_frame, ax=ax)

    sns.set(font='CMU Serif', font_scale=2)
    plt.setp(ax.get_legend().get_title(), fontsize='10')

    ax.set(ylabel='')


    #sns.legend(['minpoints = 2', 'minpoints = 3', 'minpoints = 4'], loc = 'lower right')

    c_data = create_dbscan_clusters(t_data, DBSCAN_MINPTS, max_epsilon)
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
    t_data = t_data[t_data['Time'].isin(t_data['Time'].unique()[:NO_TIMEPOINTS])]
    style.use('seaborn-paper')
    sns.set(font='CMU Serif', font_scale=1.4)
    t_data = normalize(t_data)
    for tp in range(0, NO_TIMEPOINTS):
        timepoint_data = t_data[t_data['Time']==t_data['Time'].unique()[tp]]
        timepoint_clusterings.append(list())
        for k in range(1, 6):
            c_data = create_kmeans_clusters(timepoint_data, k)
            columns = ['ObjectID', 'Time', 'cluster', 'Distance', 'ArrDelay']
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
    t_data = t_data[t_data['Time'].isin(t_data['Time'].unique()[:NO_TIMEPOINTS])]
    t_data = normalize(t_data)
    for tp in range(0, NO_TIMEPOINTS):
        timepoint_data = t_data[t_data['Time']==t_data['Time'].unique()[tp]]
        timepoint_clusterings.append(list())
        for k in range(1, 6):
            c_data = create_kmeans_clusters(timepoint_data, k)
            columns = ['ObjectID', 'Time', 'cluster', 'Distance', 'ArrDelay']
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





start_DBSCAN_experiment()