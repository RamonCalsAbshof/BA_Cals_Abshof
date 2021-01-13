import pandas as pd
import numpy as np
import math
from scipy import spatial

class PatternDetector:

    _tolerance = 0
    _comparison_method = ''

    def __init__(self,  comparison_method, tolerance):
        '''
        input
        tolerance - float which describes the tolerance depending on the comparison_method
        comparison_method - string which describes the comparison method of vectors
                            either 'manhattan', 'cosine' or 'euclidean'
        '''
        self._tolerance = tolerance
        self._comparison_method = comparison_method
        print('pattern_detection initiated')


    def calc_pattern_matrix(self, data, object_ids, feature_list):
        '''
        Calculates the pattern matrix according to the algorithm
        input
        data - dataframe which holds all time series
        object_ids - list of object ids which are compared (initially 2 only!)
        feature_list - list of features (feature_renames!)

        output
        pattern_matrix - shape (num_timestamps)^features
        '''

        # calculate the distance pattern
        distance_pattern = list()
        for timestamp in data[data['ObjectID']==object_ids[0]]['Time'].unique():
            distance = 0
            for feature in feature_list:
                sub_dist = 0
                for object_id in object_ids:
                     ts_data = (data[(data['ObjectID'] == object_id) & (data['Time'] == timestamp)][feature]).to_numpy()[0]
                     sub_dist = sub_dist - ts_data
                distance = distance + (sub_dist ** 2)
            distance_pattern.append(np.sqrt(distance))

        pattern_matrix = list()
        for i in range(0,len(distance_pattern)):
            pattern_matrix.append(list())
            pattern_matrix[i].append(distance_pattern[i])

        for i in range(0, len(distance_pattern)-1):
            for j in range(i+1, len(distance_pattern)):
                pattern_matrix[j].append(np.abs(pattern_matrix[j-1][len(pattern_matrix[j-1])-1] -
                                                pattern_matrix[j][len(pattern_matrix[j])-1]))

        return pattern_matrix

    def in_dict_list(self, dictionary, check):
        '''
        Checks whether check is a value or a key of dictionary
        input
            dictionary - arbitrary dictionary
            check - some value which is suspected to be a key or value of the dictionary
        output
            true if check is a key or value of dictionary, false if its not
        '''
        if check in dictionary.keys():
            return True
        for l in dictionary:
            if check in dictionary[l]:
                return True
        return False


    def manhattan_relative_comparison(self, sample, comp):
        '''
        The relative manhattan comparison creates a lower and an upper bound for every dimension
        of the sample. If the according dimensions of comp are within these bounds, a match is found.
        input
            sample - list of values
            comp - list of values
        output
            True if the relative manhattan distance is lower than the tolerance
        '''
        match = False
        t = self._tolerance
        for i in range(0, len(sample)):
            higher_bound = sample[i] + (sample[i] * t)
            lower_bound = sample[i] - (sample[i] * t)
            if (comp[i] < higher_bound) and (comp[i] > lower_bound):
                match = True
            else:
                match = False
                break
        return match


    def euclid_comparison(self, sample, comp):
        '''
         Calculates the euclidean distance between sample and comp. True is returned iff the euclidean distance
         is lower or equal to the user specified tolerance.
         input
             sample - list of values
             comp - list of values
         output
            Calculates the euclidean distance between sample and comp. True is returned iff the euclidean distance
            is lower or equal to the user specified tolerance.
         '''
        match = False
        t = self._tolerance
        d = 0
        for i in range(0, len(sample)):
            d = d + (sample[i]-comp[i]) ** 2
        d = math.sqrt(d)
        if d <= t:
            match = True
        return match


    def manhattan_comparison(self, sample, comp):
        '''
        Calculates the manhattan distance between sample and comp. True is returned iff the manhattan distance
        is lower or equal to the user specified tolerance.
        input
            sample - list of values
            comp - list of values
        output
            Calculates the manhattan distance between sample and comp. True is returned iff the manhattan distance
            is lower or equal to the user specified tolerance.
        '''
        match = False
        t = self._tolerance
        for i in range(0, len(sample)):
            higher_bound = sample[i] + t
            lower_bound = sample[i] - t
            if (comp[i] < higher_bound) and (comp[i] > lower_bound):
                match = True
            else:
                match = False
                break
        return match


    def cosine_comparison(self, sample, comp):
        t = self._tolerance
        d = spatial.distance.cosine(sample,comp)
        if d <= t:
            return True
        return False


    def pattern_comparison(self, sample, candidate):
        cm = self._comparison_method
        if (cm == 'euclid'):
            return self.euclid_comparison(sample, candidate)
        elif (cm == 'manhattan'):
            return self.manhattan_comparison(sample, candidate)
        elif (cm == 'cosine'):
            return self.cosine_comparison(sample, candidate)






    def find_paterns(self, pattern_matrix):
        '''
        input
        pattern_matrix - calculated pattern matrix
        output
        dictionary of patterns {[sample1]: [ [origin start, target start, length], ... ],
        [sample2]: [ [origin start, target start, length], ... ],....}
        '''

        matches = dict()
        for psize in range(2, len(pattern_matrix[len(pattern_matrix)-1])):
            for sample_time in range(psize-1, len(pattern_matrix)-1):
                for sample_idx in range(0, len(pattern_matrix[sample_time])-psize+1):
                    sample = pattern_matrix[sample_time][sample_idx:sample_idx+psize]

                    if not self.in_dict_list(matches, str(sample)):
                        matches[str(sample)]=list()
                        #get comparison sample
                        for comp_time in range(sample_time+1, len(pattern_matrix)):
                            for comp_index in range(0, len(pattern_matrix[comp_time])-psize):
                                comp = pattern_matrix[comp_time][comp_index:comp_index+psize]

                                #actual comparison
                                match = self.pattern_comparison(sample,comp)

                                if match == True:
                                    matches[str(sample)].append([sample_time, comp_time, psize])
                                    print("Sample:" + str(sample))
                                    print("Match:" + str(comp))

        results = dict()
        for match in matches.keys():
            if len(matches[match]) > 0:
                results[match] = matches[match]
        return results













