from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

class DBScan(object):

    # Global Vars
    DATA = ''
    MinPoints = 0
    EPSILON = 0
    SETTINGS = ''
    # Common Settings
    DBSCAN_METRIC = 'euclidean'
    DBSCAN_METRIC_PARAMS = None
    DBSCAN_ALGORITHM = 'auto'
    DBSCAN_LEAF_SIZE = 30
    DBSCAN_P = None
    DBSCAN_N_JOBS = None
    

    def __init__(self, data, minpts, eps, settings):
        self.DATA = data
        self.MinPts = minpts
        self.EPSILON = eps
        self.SETTINGS = settings
        
    def create_clusters(self):
        clusters=pd.DataFrame()
        cluster_count = 0
        for timestamp in self.DATA['Time'].unique() :
            timepoint_data = self.DATA.loc[self.DATA['Time'] == timestamp].copy()
    
    
            cluster_data = [tuple(x) for x in timepoint_data[self.SETTINGS['features']].values]
    
            clustering = DBSCAN(self.EPSILON, self.MinPts, self.DBSCAN_METRIC,
                                self.DBSCAN_METRIC_PARAMS, self.DBSCAN_ALGORITHM,
                                self.DBSCAN_LEAF_SIZE, self.DBSCAN_N_JOBS).fit(cluster_data)
    

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
