import numpy as np
from numba import jit


@jit
def e_p(u_x, u_y):
    ep = 0
    for i in range(0, u_x.shape[0]):
        ep = ep + np.absolute(u_x[i] - u_y[i])
    return 1 - ep/2


@jit
def hr_distance(u, v):
    u = np.array(u)
    v = np.array(v)
    sum = 0
    for i in range(0, u.shape[1]):
        for j in range(i+1, u.shape[1]):
            sum = sum + np.absolute(e_p(u[:, i], u[:, j]) - e_p(v[:, i], v[:, j]))
            #print('part sum', e_p(u[:, i], u[:, j]), e_p(v[:, i], v[:, j]), np.absolute(e_p(u[:, i], u[:, j]) - e_p(v[:, i], v[:, j])))
    #print('sum', sum)
    return (2*sum)/(u.shape[1]*(u.shape[1] - 1))


def hr_index(u, v):
    return 1 - hr_distance(u, v)