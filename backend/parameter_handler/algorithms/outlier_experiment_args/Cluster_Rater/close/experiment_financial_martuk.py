import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pymongo import MongoClient
import itertools
import urllib.parse
import matplotlib.style as style

## DBSCAN Settings
from Cluster_Rater.close.close import CLOSE

DBSCAN_METRIC = 'euclidean'
DBSCAN_METRIC_PARAMS = None
DBSCAN_ALGORITHM = 'auto'
DBSCAN_LEAF_SIZE = 30
DBSCAN_P = None
DBSCAN_N_JOBS = None
DBSCAN_MINPTS = 2

NO_TIMEPOINTS = 6
NO_COMPANIES =  60
EVAL_FUNCTION = 'mse'


########### Database Connection ##############
client = MongoClient('134.99.112.190',
                     username = 'read_user',
                     password = 'tepco11x?z',
                     authSource = 'finfraud3',
                     authMechanism = 'SCRAM-SHA-1',
                     port = 27017)

db = client.finfraud3.original
i = 0
series_list = list()
for document in db.find():
    if i == NO_COMPANIES:
        break
    else:
        i = i + 1
    if document['ric'] != 'nan':
        for year in range(1998,2017,1):
            try:
                if ((document[str(year)]['TR-NetSales'] != "nan") and (document[str(year)]['TR-NetSales'] != '') and (document[str(year)]['TR-NetSales'] != '0')
                    and (document[str(year)]['TR-TtlPlanExpectedReturn'] != "nan") and (document[str(year)]['TR-TtlPlanExpectedReturn'] != '') and (document[str(year)]['TR-TtlPlanExpectedReturn'] != '0')
                    and (document[str(year)]['TR-TotalAssetsReported'] != "nan") and (document[str(year)]['TR-TotalAssetsReported'] != '') and (document[str(year)]['TR-TotalAssetsReported'] != '0')):
                        ric = document['ric']
                        feature_1 = float(document[str(year)]['TR-TtlPlanExpectedReturn']) / float(document[str(year)]['TR-TotalAssetsReported'])
                        feature_2= float(document[str(year)]['TR-NetSales']) / float(document[str(year)]['TR-TotalAssetsReported'])
                        series_list.append(pd.Series([ric, str(year), feature_1, feature_2], index=['ObjectID', 'Time', 'ttlplanExpectedReturn', 'NetSales']))

                        print(i)
            except ZeroDivisionError :
                print('divided by zero')


t_data = pd.DataFrame(series_list, columns=['ObjectID', 'Time', 'ttlplanExpectedReturn', 'NetSales'])
t_data['Time'] = pd.to_numeric(t_data['Time'])

tmp_data = t_data.groupby(['ObjectID', 'Time']).mean().reset_index()
del t_data
t_data = tmp_data

def create_dbscan_clusters(data, minpts, eps):
    clusters=pd.DataFrame()
    cluster_count = 0
    cluster_labels = list()
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
            cluster_count =  np.max(clustering.labels_) + 1
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
        cluster_count = max(clustering.labels_)+1
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
    plot_data.rename(columns={'Time' : 'year'}, inplace=True)
    sns.set(font='CMU Serif')
    style.use('seaborn-paper')
    sns.set(font='CMU Serif', font_scale=1.4)
    #sns.set_style("darkgrid", {'axes.grid': True, 'axes.edgecolor': 'white', 'grid.color': '.9','axes.edgecolor': '.9',}, )
    g = sns.FacetGrid(plot_data, col="year", hue="cluster", col_wrap = 3, palette='Set1')
    sns.set(font='CMU Serif', font_scale=2.6)
    def f(x, y, z, w, **kwargs):
        sns.set(font_scale=2.6)
        ax = sns.scatterplot(x, y, **kwargs)
        ax.set(ylim=(-0.2,1.1))
        ax.set(xlim=(-0.2,1.1))
        ax.set(xlabel='TR-TtlPlanExpectedReturn', ylabel='TR-NetSales')
        # Individual Colors
        # For every time point use 5 different colors (assume only max 4 clusters + outliers per time point)
        # Use a palette for color blind people https://davidmathlogic.com/colorblind/
        cluster_color_palette = ['#1E88E5', '#FFC107', '#9C9C6E', '#1CF2AA', '#92C135', '#1033EA', '#55A1D7', '#38CA48', '#637645', '#81019B', '#E9E2A3', '#F0A054']
        cluster_colors = list()
        cluster_color_labels = list()
        for time in plot_data['year'].unique():
            #reset used colors per time point
            color_index = 0
            for cluster_id in plot_data[plot_data['year'] == time]['cluster'].unique():
                if cluster_id != -1 :
                    cluster_colors.append(cluster_color_palette[color_index])
                    cluster_color_labels.append(cluster_id)
                    color_index = color_index + 1

        cluster_colors.append('#D81B60')
        cluster_color_labels.append(-1)

        c = lambda x: cluster_colors[cluster_color_labels.index(x)]

        for i in range(len(x)):
            ax.annotate(z.values[i], xy=(x.values[i], y.values[i]), fontsize=12,
                        xytext=(0, 0), textcoords="offset points",
                        color='black',
                        bbox=dict(boxstyle="round", alpha=0.3, color=c(w.values[i])),
                        va='center', ha='center', weight='bold', alpha=1)
           # print(labels[z.values[i]]+' ' + str(w.values[i]) + str(kwargs.get("color", "k")))

    g.map(f, "ttlplanExpectedReturn", "NetSales", "ObjectID", "cluster", alpha=0.6, s=5, legend='full')
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle(title)
    sns.set(font_scale=2.6)
    return plt



def normalize(data):
    Distance = data[['ttlplanExpectedReturn']].values.astype(float)
    arrdelay = data[['NetSales']].values.astype(float)

    min_max_scaler_dep = preprocessing.MinMaxScaler()
    dep_scaled = min_max_scaler_dep.fit_transform(Distance)

    min_max_scaler_arr = preprocessing.MinMaxScaler()
    arr_scaled = min_max_scaler_arr.fit_transform(arrdelay)

    data['ttlplanExpectedReturn'] = dep_scaled
    data['NetSales'] = arr_scaled

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
    t_data = t_data[t_data['Time'].isin(t_data['Time'].unique()[10:NO_TIMEPOINTS + 10])]
    style.use('seaborn-paper')
    sns.set(font='CMU Serif', font_scale=4)
    fig, ax = plt.subplots()
    for minpts in np.arange(2, 5, 1):
        eps_list = []
        scores_list = []
        stability_list = []
        mse_list = []
        minpts_list = []
        for epsilon in np.arange(0.05, 0.25, 0.01):
            c_data = normalize(t_data)
            c_data = create_dbscan_clusters(c_data, minpts, epsilon)
           # plot = plotTimePoints(c_data, NO_TIMEPOINTS)
            columns = ['ObjectID', 'Time', 'cluster', 'ttlplanExpectedReturn', 'NetSales']
            c_data = c_data[columns]
            c_data = normalize(c_data)
            rater = CLOSE(c_data, EVAL_FUNCTION, DBSCAN_MINPTS)
            scores = rater.rate_clustering(return_measures=True)
            clustering_score = scores['close']
            quality_score = scores['quality']
            stability_score = scores['stability']

            if float(clustering_score) > max_score:
                max_score = clustering_score
                print('New MAX Leader' + str(epsilon))
                max_epsilon = epsilon
                max_minpts = minpts

            if float(clustering_score) < 0:
                print('hallo')

            print('Epsilon: ' + str(epsilon))
            print('Total Clustering Score: ', str(clustering_score))
            #plot.show()
            eps_list.append(epsilon)
            scores_list.append(clustering_score)


        d = {'epsilon' : eps_list, 'score' : scores_list}
        d_frame = pd.DataFrame(d)
        sns.set(font='CMU Serif', font_scale=5)
        g = sns.lineplot(x='epsilon', y='score', markers=True, label = str(minpts), data=d_frame, ax = ax)
        g.set(xticks=[0.05, 0.1, 0.15, 0.2, 0.25])

    ax.legend().set_title('MinPts')
    plt.setp(ax.get_legend().get_texts(), fontsize='40')
    plt.setp(ax.get_legend().get_title(), fontsize='40')

    sns.set(font='CMU Serif', font_scale=5)



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
            columns = ['ObjectID', 'Time', 'cluster', 'ttlplanExpectedReturn', 'NetSales']
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