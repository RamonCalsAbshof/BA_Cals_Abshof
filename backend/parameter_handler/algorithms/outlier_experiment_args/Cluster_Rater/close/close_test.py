import pandas
from Cluster_Rater.close import CLOSE
from Cluster_Rater.close_v2 import CLOSEv2
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt


def calc_dev_data(data):
    columns = data.columns
    feature_columns = columns[3:]
    objects = data[columns[0]].unique()
    timestamps = data[columns[1]].unique()
    new_rows = []
    for object in objects:
        for i in range(len(timestamps) - 1):
            if len(data[(data[columns[0]] == object) & (data[columns[1]] == timestamps[i])]) > 0 \
                    and len(data[(data[columns[0]] == object) & (data[columns[1]] == timestamps[i + 1])]) > 0:
                row = data[(data[columns[0]] == object) & (data[columns[1]] == timestamps[i])]
                new_row = row[columns[:3]].iloc[0].tolist()
                for feature in feature_columns:
                    dev = data[(data[columns[0]] == object) & (data[columns[1]] == timestamps[i])][feature].item() - \
                          data[(data[columns[0]] == object) & (data[columns[1]] == timestamps[i + 1])][feature].item()
                    new_row.append(dev)
                new_rows.append(new_row)
    dev_data = pandas.DataFrame(new_rows, columns=columns)
    return dev_data


if __name__ == '__main__':
    # (min-max) normalised data
    test_data = [[1, 1, 1, 0.2, 0.167], [2, 1, 1, 0.4, 0.167], [3, 1, 1, 0.2, 0.33], [4, 1, 2, 0.4, 0.66], [5, 1, 2, 0.6, 0.66], [6, 1, 2, 0.4, 0.83], [7, 1, 2, 0.6, 0.83],
                 [1, 2, 3, 0.4, 0.167], [2, 2, 3, 0.6, 0.167], [3, 2, 3, 0.4, 0.33], [4, 2, 4, 0.4, 0.83], [5, 2, 3, 0.6, 0.33], [6, 2, 4, 0.4, 1.0], [7, 2, -1, 1.0, 1.0],
                 [1, 3, 6, 0.4, 0.167], [2, 3, 6, 0.4, 0.33], [3, 3, 6, 0.2, 0.167], [4, 3, 7, 0.4, 0.83], [5, 3, 8, 0.8, 0.5], [6, 3, 7, 0.2, 1.0], [7, 3, 7, 0.4, 0.83]]

    # # one big cluster (normalised)
    # test_data = [[1, 1, 1, 1/3, 1/6], [2, 1, 1, 2/3, 1/6], [3, 1, 1, 1/3, 2/6], [4, 1, 1, 2/3, 4/6], [5, 1, 1, 3/3, 4/6], [6, 1, 1, 2/3, 5/6],
    #              [1, 2, 3, 2/3, 1/6], [2, 2, 3, 3/3, 1/6], [3, 2, 3, 2/3, 2/6], [4, 2, 3, 2/3, 5/6], [5, 2, 3, 3/3, 5/6], [6, 2, 3, 2/3, 6/6],
    #              [1, 3, 6, 2/3, 1/6], [2, 3, 6, 2/3, 2/6], [3, 3, 6, 1/3, 1/6], [4, 3, 6, 2/3, 5/6], [5, 3, 6, 3/3, 4/6], [6, 3, 6, 1/3, 6/6]]
    #
    # # two perfect clusters (normalised)
    # test_data = [[1, 1, 1, 1/3, 1/6], [2, 1, 1, 2/3, 1/6], [3, 1, 1, 1/3, 2/6], [4, 1, 2, 2/3, 4/6], [5, 1, 2, 3/3, 4/6], [6, 1, 2, 2/3, 5/6],
    #              [1, 2, 3, 2/3, 1/6], [2, 2, 3, 3/3, 1/6], [3, 2, 3, 2/3, 2/6], [4, 2, 4, 2/3, 5/6], [5, 2, 4, 3/3, 5/6], [6, 2, 4, 2/3, 6/6],
    #              [1, 3, 6, 2/3, 1/6], [2, 3, 6, 2/3, 2/6], [3, 3, 6, 1/3, 1/6], [4, 3, 7, 2/3, 5/6], [5, 3, 7, 3/3, 4/6], [6, 3, 7, 1/3, 6/6]]
    #
    # # two perfect clusters
    # test_data = [[1, 1, 1, 1, 1], [2, 1, 1, 2, 1], [3, 1, 1, 1, 2], [4, 1, 2, 2, 4], [5, 1, 2, 3, 4], [6, 1, 2, 2, 5],
    #              [1, 2, 3, 2, 1], [2, 2, 3, 3, 1], [3, 2, 3, 2, 2], [4, 2, 4, 2, 5], [5, 2, 4, 3, 5], [6, 2, 4, 2, 6],
    #              [1, 3, 6, 2, 1], [2, 3, 6, 2, 2], [3, 3, 6, 1, 1], [4, 3, 7, 2, 5], [5, 3, 7, 3, 4], [6, 3, 7, 1, 6]]


    cl = pandas.DataFrame(test_data, columns=['object_id', 'time', 'cluster_id', 'feature1', 'feature2'])
    # rater = CLOSE(cl, 'mae', 2, jaccard=True, weighting=True)
    rater = CLOSEv2(cl, 'mae', 2)
    clustering_score_mu = rater.rate_clustering()

    rand_index = rater.calc_rand_rating()
    print('Average Rand Index: ', str(rand_index))

    print('Total CLOSE Score: ', str(clustering_score_mu))
