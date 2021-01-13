import numpy as np

def similarity_matrix(u, v):
    u = np.array(u)
    v = np.array(v)

    k = u.shape[0]
    l = v.shape[0]
    s = np.zeros((k, l))
    for i in range (0, k):
        for j in range (0, l):
            s[i][j] = np.minimum(u[i], v[j]).sum()/np.maximum(u[i], v[j]).sum()
    return s



def subset_similarity(u, v):
    s = similarity_matrix(u, v)
    subset_measure = np.amin(np.amax(s, axis=1))
    superset_measure = np.amin(np.amax(s, axis=0))
    return max(subset_measure, superset_measure)



