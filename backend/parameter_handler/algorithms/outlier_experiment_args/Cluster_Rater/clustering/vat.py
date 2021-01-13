import numpy as np


def vat(dist_matrix):
    dist_copy = dist_matrix.copy()
    set_I = []
    set_J = list(range(dist_matrix.shape[0]))
    set_P = np.zeros((dist_matrix.shape[0],), dtype=np.int)
    j = int(dist_matrix.argmax()/dist_matrix.shape[0])
    set_P[0] = j
    set_I.append(j)
    set_J.remove(j)
    # fill diagonal with max values so that it isn't a minimum
    np.fill_diagonal(dist_matrix, np.finfo(np.float64).max)
    for t in range(1, dist_matrix.shape[0]):
        # look only in the rows of set_I
        i = dist_matrix[np.ix_(set_I)].argmin() % dist_matrix.shape[0]
        set_P[t] = i
        # fill values of d with indexes set_I x set_I with maximum
        dist_matrix[np.ix_(set_I, [i])] = np.finfo(np.float64).max
        dist_matrix[np.ix_([i], set_I)] = np.finfo(np.float64).max
        set_I.append(i)
        set_J.remove(i)
    d_vat = np.zeros((dist_matrix.shape[0], dist_matrix.shape[0]))
    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[0]):
            d_vat[i][j] = dist_copy[set_P[i]][set_P[j]]
    return d_vat


