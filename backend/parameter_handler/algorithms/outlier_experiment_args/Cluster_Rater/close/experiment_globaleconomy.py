import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import itertools
from sqlalchemy import create_engine
import sqlalchemy as sql
import matplotlib.style as style
## DBSCAN Settings
from Cluster_Rater.close.close import CLOSE
from Cluster_Rater.data_plotter import DataPlotter

DBSCAN_METRIC = 'euclidean'
DBSCAN_METRIC_PARAMS = None
DBSCAN_ALGORITHM = 'auto'
DBSCAN_LEAF_SIZE = 30
DBSCAN_P = None
DBSCAN_N_JOBS = None
KMEANS_SEED = 7

data_table = sql.table('data')
column2_is_percent = True
column1_is_percent = True
column1 = '10' # Unemployment rate
column2 = '150' # Household consumption percent of GDP
column1_name = 'Unemployment Rate'
column2_name = 'Edu. Spendings'
start_year = 2010 #included
end_year = 2015 #included
NO_TIMEPOINTS = end_year - start_year +1

SETTINGS = {'features' : [column1, column2],
          'feature_renames' : [column1_name, column2_name],
          'start_time' : start_year,
          'end_time' : end_year}



########### Database Connection ##############
engine = create_engine("mysql://globaleconomy:rocrame@134.99.112.190:27020/globaleconomy2", echo=False)
connection = engine.connect()
s  = connection.execute('SELECT `Code`, `Year`, `Month`, `'+column1+'`, `'+column2+
                        '` FROM data WHERE Month = 12 AND Year>='+str(start_year) + ' AND Year<='+str(end_year)+\
     ' AND `' + column1 + '` IS NOT NULL AND `'+ column2 + '` IS NOT NULL ORDER BY `CODE` LIMIT 250,120')





t_data = pd.DataFrame(s, columns=['ObjectID', 'Year', 'Month', column1, column2])



t_data['Time'] = t_data['Year'].map(str)+ '-' +t_data['Month'].map(str) +'-01 13:00:00'

t_data = t_data[['ObjectID', 'Time', column1, column2]]

t_data['Time'] = pd.to_datetime(t_data['Time'])
t_data['Time'] = t_data['Time'].astype(np.int64)/1000000000
t_data['Time'] = t_data['Time'].astype(int)
t_data = t_data.rename(columns={column1: column1_name, column2 : column2_name})
t_data = t_data.reset_index()

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

        clustering = KMeans(n_clusters=n, random_state=KMEANS_SEED).fit(cluster_data)

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
    data['Time'] = pd.to_datetime(data['Time'], unit='s')
    data['Time'] = data['Time'].map(lambda x: x.year)
    data['Year'] = data['Time']
    data = data[['ObjectID','Year', column1_name, column2_name, 'cluster']]
    plot_data = data[data['Year'].isin(data['Year'].unique()[:no])]

    style.use('seaborn-paper')
    sns.set(font='CMU Serif', font_scale=1.6)
    g = sns.FacetGrid(plot_data, col="Year", hue="cluster", col_wrap = 4, palette='Set1')
    def f(x, y, z, w, **kwargs):
        ax = sns.scatterplot(x, y, **kwargs)
        ax.set(ylim=(-0.1,1.1))
        ax.set(xlim=(-0.1,1.1))
        for i in range(len(x)):
            ax.annotate(z.values[i], xy=(x.values[i], y.values[i]), fontsize=10,
                        xytext=(0, 0), textcoords="offset points",
                        color='black',
                        bbox=dict(boxstyle="round", alpha=0.3, color=kwargs.get("color", "k")),
                        va='center', ha='center', weight='bold', alpha=1)
           # print(labels[z.values[i]]+' ' + str(w.values[i]) + str(kwargs.get("color", "k")))

    g.map(f, column1_name, column2_name, "ObjectID", "cluster", alpha=0.6, s=5, legend='full')
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle(title)
    return plt


def normalize(data):
   # data[column1_name] = data[column1_name].astype(float)/100.0
    #data[column2_name] = data[column2_name].astype(float)/100.0



    if float(data[column1_name].max()) > 1.0 or  float(data[column1_name].min()) < 0.0:
        f1 = data[[column1_name]].values.astype(float)
        min_max_scaler_dep = preprocessing.MinMaxScaler()
        dep_scaled = min_max_scaler_dep.fit_transform(f1)
        data[column1_name] = dep_scaled


    if float(data[column2_name].max()) > 1.0 or float(data[column2_name].min()) < 0.0:
        f2 = data[[column2_name]].values.astype(float)
        min_max_scaler_arr = preprocessing.MinMaxScaler()
        arr_scaled = min_max_scaler_arr.fit_transform(f2)
        data[column2_name] = arr_scaled



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
    for k in range(1, 10, 1):
        t_data = t_data[t_data['Time'].isin(t_data['Time'].unique()[:NO_TIMEPOINTS])]
        c_data = normalize(t_data)
        c_data = create_kmeans_clusters(t_data, k)
        # plot = plotTimePoints(c_data, NO_TIMEPOINTS)
        columns = ['ObjectID', 'Time', 'cluster', column1_name, column2_name]
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
    sns.set(font='CMU Serif', font_scale=5, rc={'figure.figsize':(12, 6)})
    ax = sns.lineplot(x='k', y='score', hue='variable', data=pd.melt(d_frame, ['k'], value_name='score'), dashes=True)
    ax.set(xticks=np.arange(1,11,1))
    ax.set(yticks=[0.0,0.5,1.0])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], fontsize='35')
    sns.set(font='CMU Serif', font_scale=5)
    plt.gcf().subplots_adjust(bottom=0.2)


    print('MAX, MIN, MID ' + str(max_k) + ' ' + str(min_k) + ' ' + str(mid_k))

    c_data = create_kmeans_clusters(t_data, max_k)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'k='+str(max_k))
    plot.savefig('images/'+str(max_k)+'_k_kmeans_globaleconomy.png', transparent=True)
    plot.show()

    c_data = create_kmeans_clusters(t_data, 4)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'k='+str(4))
    plot.savefig('images/'+str(min_k) + '_k_kmeans_globaleconomy.png', transparent=True)
    plot.show()

    c_data = create_kmeans_clusters(t_data, 3)
    plot = plotTimePoints(c_data, NO_TIMEPOINTS, 'k='+str(3))
    plot.savefig('images/'+str(mid_k) + '_k_kmeans_globaleconomy.png', transparent=True)
    plot.show()
    sns.set(font='CMU Serif', font_scale=5)
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
    #t_data = t_data[t_data['Time'].isin(t_data['Time'].unique()[10:NO_TIMEPOINTS+10])]
    style.use('seaborn-paper')
    sns.set(font='CMU Serif', font_scale=1.4)
    fig, ax = plt.subplots()
    for epsilon in np.arange(0.05, 1.5, 0.05):
        c_data = normalize(t_data)
        c_data = create_dbscan_clusters(t_data, 2, epsilon)
       # plot = plotTimePoints(c_data, NO_TIMEPOINTS)
        columns = ['ObjectID', 'Time', 'cluster', column1_name, column2_name]
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
            columns = ['ObjectID', 'Time', 'cluster', column1_name, column2_name]
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


def plot_clustering(method='dbscan', minPts=2, epsilon=0.19, k=5):
    style.use('seaborn-paper')
    sns.set(font='CMU Serif', font_scale=1.4)
    global t_data
    c_data = normalize(t_data)
    if method == 'dbscan':
        c_data = create_dbscan_clusters(c_data, minPts, epsilon)
    elif method == 'kmeans':
        c_data = create_kmeans_clusters(c_data, k)
    else:
        print('Unknown method name.')
        return

    plotter = DataPlotter(SETTINGS)
    plot = plotter.plot_twodim_clusters(c_data, outlier=False)
    plot.show()





# start_KMEANS_experiment()
# start_DBSCAN_experiment()
plot_clustering(method='dbscan', minPts=2, epsilon=0.19)