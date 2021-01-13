import numpy as np
from scipy.spatial.distance import cdist
from Cluster_Rater.fuzzy.normalize_columns import normalize_columns, normalize_power_columns
import random
from numpy import genfromtxt

def distance_np(data, centers):
    return cdist(data, centers)

def fuzzy_c_means_0(data, centers, m):
    # step 1: membership matrix calculation
    dist = distance_np(data, centers).T
    dist = np.fmax(dist, np.finfo(np.float64).eps)
    u = normalize_power_columns(dist, - 2. / (m - 1))
    u = np.fmax(u, np.finfo(np.float64).eps)

    # step 2: cluster centers calculation
    um = u ** m
    centers_new = um.dot(data) / (np.ones((data.shape[1],1)).dot(np.atleast_2d(um.sum(axis=1))).T)

    return centers_new, u

def initialize_centers(data, c):
    centers = np.empty((c, data.shape[1]))
    for i in range(0, c):
        for j in range(0, data.shape[1]):
            centers[i][j] = random.uniform(np.amin(data[:,j]), np.amax(data[:,j]))
    return centers

def fcm(data, c, m, error, maxiter, init_cntr=None):

    if init_cntr == None:
        init_cntr = initialize_centers(data, c)
    centers_new = init_cntr

    number_of_loops = 0

    # main loop
    while number_of_loops < maxiter:
        centers = centers_new.copy()
        [centers_new, u_new] = fuzzy_c_means_0(data, centers, m)
        number_of_loops = number_of_loops + 1

        if np.linalg.norm(centers_new - centers) < error:
            break

    # calculate jm
    dist = distance_np(data, centers_new).T
    dist = np.fmax(dist, np.finfo(np.float64).eps)
    jm = ((u_new ** m) * (dist ** 2)).sum()

    return centers_new, u_new, jm, number_of_loops

