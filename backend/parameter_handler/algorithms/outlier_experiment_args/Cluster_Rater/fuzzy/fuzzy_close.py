import os
import numpy as np
import pandas as pd
import Cluster_Rater.fuzzy.fcm_Blatt2 as cl
import Cluster_Rater.fuzzy.subset_similarity as ss
import Cluster_Rater.fuzzy.hr_index as hr
from collections import OrderedDict
import math
import Cluster_Rater.fuzzy.settings as s
import Cluster_Rater.fuzzy.mongodb_provider as mongodb

DATASET = s.financial
start_year = DATASET['start_year']
end_year = DATASET['end_year']
clusters_per_year = DATASET['clusters_per_year']
num_time_series = DATASET['num_timeseries']


def load_data():

    if DATASET['Name'] == 'globaleconomy':
        directory = 'D:\\Seafile\\workspace\\finfraud\\Cluster_Rater\\fuzzy\\data\\'
        all_data = list()
        for file in os.listdir(directory):
            if file.endswith(".csv"):

                data = pd.read_csv(directory+file, header=None)
                data = data.drop([0], axis=1)
                data = data.to_numpy()
                all_data.append(data)

        return all_data

    if DATASET['Name'] == 'financial':
        data_provider = mongodb.MongoDB(DATASET)
        data = data_provider.get_financial_data()
        # now make sure the retrieved data is complete
        final_data = pd.DataFrame(columns=data.columns)
        for object in data['ObjectID'].unique():
            if len(data[data['ObjectID']==object]) == (end_year - start_year + 1):
                final_data = final_data.append(data[data['ObjectID']==object])

        all_data = list()
        for year in data['Time'].unique():
            tmp_data = final_data[data['Time']==year].copy()
            tmp_data = tmp_data.drop(['ObjectID', 'Time'], axis = 1)
            tmp_data = tmp_data.to_numpy()
            all_data.append(tmp_data)
        return all_data









def cluster_data(all_data):
    clustering_results = dict()
    clustering_results['c'] = dict()
    clustering_results['u'] = dict()
    for year in range(start_year, end_year+1):
        clustering_results['c'][str(year)] = clusters_per_year[str(year)]
    counter = start_year
    for data in all_data:
        centers_new, u_new, jm, number_of_loops = cl.fcm(data, clustering_results['c'][str(counter)], 2.0, 0.0001, 1000)
        clustering_results['u'][str(counter)] = u_new
        counter = counter + 1

    return clustering_results



def calc_subset_sim(clustering_results):
    sim_matrix = np.zeros(((end_year - start_year),(end_year - start_year)))
    for i in range(start_year, end_year):
        for j in range(start_year, end_year):
            print(i)
            print(j)
            sim_matrix[i-start_year][j-start_year] = ss.subset_similarity(clustering_results['u'][str(i)],clustering_results['u'][str(j)])
    overall_similarity = sim_matrix.sum()
    overall_similarity = ((overall_similarity - 7.0) / 2 ) / 21
    print('overall: ' + str(overall_similarity))


    return sim_matrix

#out of place distance of two dicts (regarding their keys)
def oop_distance(a, b):
    distance = 0
    for k in a.keys():
        distance = distance + math.fabs(list(a.keys()).index(k) - list(b.keys()).index(k))
    return distance / (len(list(a.keys())))


def calc_sequence_stability(clustering_results):
    hr_dict = dict()

    # Calculate e_p for every pair of time series
    for i in range(start_year, end_year):
        hr_dict[str(i)] = dict()
        for j in range(0, num_time_series-1):
            hr_dict[str(i)][j] = OrderedDict()
            for k in range(0, num_time_series-1):
                #tmp = clustering_results['u'][str(i)][:,j]
                hr_dict[str(i)][j][k] = hr.e_p(clustering_results['u'][str(i)][:,j], clustering_results['u'][str(i)][:,k])
            # Sort dicts by e_p
            sorted_dict = OrderedDict({k: v for k, v in sorted(hr_dict[str(i)][j].items(), key=lambda item: item[1], reverse=True)})
            hr_dict[str(i)][j] = sorted_dict
    # calculate out of place distance for every time series
    stability_dict = dict()
    for i in range(0, num_time_series-1):
        normalize_counter = 0
        oop_sum = 0
        for y1 in range(start_year, end_year):
            for y2 in range(start_year, end_year):
                if y2 > y1:
                    oop_sum = oop_sum + oop_distance(hr_dict[str(y1)][i], hr_dict[str(y2)][i])
                    normalize_counter = normalize_counter + 1
        stability_dict[i] = oop_sum / normalize_counter

    print(stability_dict)






all_data = load_data()
clustering_results = cluster_data(all_data)
calc_sequence_stability(clustering_results)
sim_matrix = calc_subset_sim(clustering_results)



