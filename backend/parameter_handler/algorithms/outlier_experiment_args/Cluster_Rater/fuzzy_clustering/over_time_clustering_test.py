from Cluster_Rater.clustering.over_time_clustering import OverTimeClustering
import pandas as pd


def test():
    test_data = pd.DataFrame({'ObjectID': ['a', 'b', 'c', 'd', 'e', 'f',
                                           'a', 'b', 'c', 'd', 'e', 'f',
                                           'a', 'b', 'c', 'd', 'e', 'f',
                                           'a', 'b', 'c', 'd', 'e', 'f',
                                           'a', 'b', 'c', 'd', 'e', 'f'],
                              'Time': [1, 1, 1, 1, 1, 1,
                                       2, 2, 2, 2, 2, 2,
                                       3, 3, 3, 3, 3, 3,
                                       4, 4, 4, 4, 4, 4,
                                       5, 5, 5, 5, 5, 5],
                              'Feature': [0.9, 0.6, 0.5, 0.4, 0.2, 0.1,
                                          0.9, 0.6, 0.5, 0.3, 0.2, 0.1,
                                          0.8, 0.6, 0.5, 0.4, 0.2, 0.1,
                                          0.9, 0.6, 0.5, 0.4, 0.3, 0.1,
                                          0.9, 0.5, 0.6, 0.4, 0.2, 0.1]})
    otc = OverTimeClustering(test_data)
    preferences = otc.get_temporal_preferences()
    print(preferences)
    preferences = otc.get_preferences_df("temporal")
    print('Ready')


test()