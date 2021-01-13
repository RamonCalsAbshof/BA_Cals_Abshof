import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from matplotlib import pyplot as plt
import sys
import ast
import random

from Cluster_Rater.data_plotter import DataPlotter
from Cluster_Rater.ts_patterns.data_config import *
from Cluster_Rater.ts_patterns.pattern_detection import *
from Cluster_Rater.helpers import *
import matplotlib.pyplot as plt

comp_objects = ['WN', 'UA']
restat_folder = 'label_files/'
restat_csv = 'restatements_audit_analytics.csv'

calc_eval = True
plot_results = False
draw_points = True
deviate = False



DATASET = flights_oned
data = load_data(DATASET)


pattern_detector = PatternDetector('euclid', 0.020)
pattern_matrix = pattern_detector.calc_pattern_matrix(data, comp_objects, DATASET['feature_renames'])
matches = pattern_detector.find_paterns(pattern_matrix)
print(matches)
print(pattern_matrix)
data = data[(data['ObjectID'].isin(comp_objects))]
for object in data['ObjectID'].unique():
    tmp = data[data['ObjectID']==object]
    plt.plot('Time', 'Distance', data=tmp)

polys = list()
for match_list in matches.keys():
    for match in matches[match_list]:
        start_start = data[(data['ObjectID'] == comp_objects[0]) & (data['Time'] == match[0])][DATASET['feature_renames']].values[0][0]
        start_end = data[(data['ObjectID'] == comp_objects[0]) & (data['Time'] == (match[0]+match[2]-1))][DATASET['feature_renames']].values[0][0]

        target_start = data[(data['ObjectID'] == comp_objects[1]) & (data['Time'] == match[1])][DATASET['feature_renames']].values[0][0]
        target_end = data[(data['ObjectID'] == comp_objects[1]) & (data['Time'] == (match[1]+match[2]-1))][DATASET['feature_renames']].values[0][0]

        coords = [[match[0], start_start], [(match[0]+match[2]-1), start_end], [(match[1]+match[2]-1), target_end], [match[1], target_start], [match[0], start_start]]
        xs,ys = zip(*coords)
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(1)][0]
        plt.fill(xs, ys, color , alpha = 0.2)

plt.show()




