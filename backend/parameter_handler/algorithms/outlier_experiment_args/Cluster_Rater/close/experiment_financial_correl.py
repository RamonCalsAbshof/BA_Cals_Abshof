import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pymongo import MongoClient
import itertools

## DBSCAN Settings
from Cluster_Rater.close.close import CLOSE

DBSCAN_METRIC = 'euclidean'
DBSCAN_METRIC_PARAMS = None
DBSCAN_ALGORITHM = 'auto'
DBSCAN_LEAF_SIZE = 30
DBSCAN_P = None
DBSCAN_N_JOBS = None

NO_TIMEPOINTS = 3


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
    if i == 30:
        break
    else:
        i = i + 1
    if document['ric'] != 'nan':
        for year in range(1998,2017,1):
            try:
                if ((document[str(year)]['TR-Employees'] != "nan") and (document[str(year)]['TR-Employees'] != '') and (document[str(year)]['TR-Employees'] != '0')
                    and (document[str(year)]['TR-Revenue'] != "nan") and (document[str(year)]['TR-Revenue'] != '') and (document[str(year)]['TR-Revenue'] != '0')
                    and (document[str(year)]['TR-TotalAssetsReported'] != "nan") and (document[str(year)]['TR-TotalAssetsReported'] != '') and (document[str(year)]['TR-TotalAssetsReported'] != '0')):
                        ric = document['ric']
                        feature_1 = float(document[str(year)]['TR-Revenue']) / float(document[str(year)]['TR-TotalAssetsReported'])
                        feature_2= float(document[str(year)]['TR-Employees']) / float(document[str(year)]['TR-TotalAssetsReported'])
                        series_list.append(pd.Series([ric, str(year), feature_1, feature_2], index=['ObjectID', 'Time', 'Revenue', 'Employees']))

                        print(i)
            except ZeroDivisionError :
                print('divided by zero')


t_data = pd.DataFrame(series_list, columns=['ObjectID', 'Time', 'Revenue', 'Employees'])
t_data['Time'] = pd.to_numeric(t_data['Time'])

tmp_data = t_data.groupby(['ObjectID', 'Time']).mean().reset_index()
del t_data
t_data = tmp_data

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
    plot_data = data[data['Time'].isin(data['Time'].unique()[:no])]
    sns.set_style("whitegrid", {'axes.grid': True, 'axes.edgecolor': 'black'})
    g = sns.FacetGrid(plot_data, col="Time", hue="cluster", col_wrap = 3, palette='Set1')
    def f(x, y, z, w, **kwargs):
        ax = sns.scatterplot(x, y, **kwargs)
        ax.set(ylim=(-0.2,1.1))
        ax.set(xlim=(-0.2,1.1))
        for i in range(len(x)):
            ax.annotate(str(i), xy=(x.values[i], y.values[i]), fontsize=8,
                        xytext=(0, 0), textcoords="offset points",
                        color='black',
                        bbox=dict(boxstyle="round", alpha=0.3, color=kwargs.get("color", "k")),
                        va='center', ha='center', weight='bold', alpha=1)
           # print(labels[z.values[i]]+' ' + str(w.values[i]) + str(kwargs.get("color", "k")))

    g.map(f, "Revenue", "Employees", "ObjectID", "cluster", alpha=0.6, s=5, legend='full')
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle(title)
    return plt



def normalize(data):
    Distance = data[['Revenue']].values.astype(float)
    arrdelay = data[['Employees']].values.astype(float)

    min_max_scaler_dep = preprocessing.MinMaxScaler()
    dep_scaled = min_max_scaler_dep.fit_transform(Distance)

    min_max_scaler_arr = preprocessing.MinMaxScaler()
    arr_scaled = min_max_scaler_arr.fit_transform(arrdelay)

    data['Revenue'] = dep_scaled
    data['Employees'] = arr_scaled

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
    global t_data
    for k in range(1, 20, 1):
        t_data = t_data[t_data['Time'].isin(t_data['Time'].unique()[:NO_TIMEPOINTS])]
        c_data = normalize(t_data)
        c_data = create_kmeans_clusters(t_data, k)
        # plot = plotTimePoints(c_data, NO_TIMEPOINTS)
        columns = ['ObjectID', 'Time', 'cluster', 'Revenue', 'Employees']
        c_data = c_data[columns]
        c_data = normalize(c_data)
        rater = CLOSE(c_data)
        clustering_score = rater.rate_clustering()

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

    d = {'k': k_list, 'score': scores_list}
    d_frame = pd.DataFrame(d)
    sns.set_style("whitegrid", {'axes.grid': True, 'axes.edgecolor': 'black'})
    sns.lineplot(x='k', y='score', data=d_frame)

    print('MAX, MIN, MID ' + str(max_k) + ' ' + str(min_k) + ' ' + str(mid_k))

    c_data = create_kmeans_clusters(t_data, max_k)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'k='+str(max_k))
    plot.show()

    c_data = create_kmeans_clusters(t_data, min_k)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'k='+str(min_k))
    plot.show()

    c_data = create_kmeans_clusters(t_data, mid_k)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'k='+str(mid_k))
    plot.show()

    c_data = create_kmeans_clusters(t_data, 3)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'k='+str(3))
    plot.show()


def start_DBSCAN_experiment():
    max_score = -2.0
    max_epsilon = 0.0
    min_score = 2.0
    min_epsion = 0.0
    mid_score = 2.0
    mid_epsilon = 0.0
    global t_data
    eps_list = []
    scores_list=[]
    t_data = t_data[t_data['Time'].isin(t_data['Time'].unique()[10:NO_TIMEPOINTS+10])]
    for epsilon in np.arange(0.05, 1.5, 0.05):
        c_data = normalize(t_data)
        c_data = create_dbscan_clusters(t_data, 2        , epsilon)
       # plot = plotTimePoints(c_data, NO_TIMEPOINTS)
        columns = ['ObjectID', 'Time', 'cluster', 'Revenue', 'Employees']
        c_data = c_data[columns]
        c_data = normalize(c_data)
        rater = CLOSE(c_data)
        clustering_score = rater.rate_clustering()

        if clustering_score > max_score:
            max_score = clustering_score
            print('New MAX Leader' + str(epsilon))
            max_epsilon = epsilon

        if clustering_score < min_score:
            min_score = clustering_score
            print('New MIN Leader' + str(epsilon))
            min_epsilon = epsilon

        if abs(clustering_score) < mid_score:
            mid_score = abs(clustering_score)
            print('New Mid Leader' + str(epsilon))
            mid_epsilon = epsilon
        print('Epsilon: ' + str(epsilon))
        print('Total Clustering Score: ', str(clustering_score))
        #plot.show()

        eps_list.append(epsilon)
        scores_list.append(clustering_score)

    d = {'epsilon' : eps_list, 'score' : scores_list}
    d_frame = pd.DataFrame(d)
    sns.set_style("whitegrid", {'axes.grid': True, 'axes.edgecolor': 'black'})
    sns.lineplot(x='epsilon', y = 'score', data = d_frame)

    print('MAX, MIN, MID ' + str(max_epsilon) + ' ' + str(min_epsilon) + ' ' + str(mid_epsilon))

    c_data = create_dbscan_clusters(t_data, 3, max_epsilon)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS,  'epsilon ='+str(max_epsilon))
    plot.show()

    c_data = create_dbscan_clusters(t_data, 3, min_epsilon)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'epsilon ='+str(min_epsilon))
    plot.show()

    c_data = create_dbscan_clusters(t_data, 3, mid_epsilon)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'epsilon = '+str(mid_epsilon))
    plot.show()

    c_data = create_dbscan_clusters(t_data, 3, 0.15)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'epsilon = 0.15')
    plot.show()

    c_data = create_dbscan_clusters(t_data, 3, 0.25)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'epsilon = 0.25')
    plot.show()

    c_data = create_dbscan_clusters(t_data, 3, 0.4)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'epsilon = 0.4')
    plot.show()

    c_data = create_dbscan_clusters(t_data, 3, 0.55)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'epsilon = 0.55')
    plot.show()

    c_data = create_dbscan_clusters(t_data, 3, 0.66)
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
            columns = ['ObjectID', 'Time', 'cluster', 'Revenue', 'Employees']
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