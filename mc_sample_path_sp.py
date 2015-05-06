import numpy as np
from scipy import sparse
from numba import jit


@jit(nopython=True)
def searchsorted(a, v):
    lo = -1
    hi = len(a)
    while(lo < hi-1):
        m = (lo + hi) // 2
        if v < a[m]:
            hi = m
        else:
            lo = m
    return hi


@jit
def mc_sample_path_sp(P, init=0, sample_size=1000):
    P = sparse.csr_matrix(P)
    n = P.shape[0]
    data = P.data
    indices = P.indices
    indptr = P.indptr

    # CDFs, one for each row of P
    cdfs1d = np.empty(P.nnz)
    for i in range(n):
        cdfs1d[indptr[i]:indptr[i+1]] = data[indptr[i]:indptr[i+1]].cumsum()

    # Random values, uniformly sampled from [0, 1)
    u = np.random.random(size=sample_size)

    # === set up array to store output === #
    X = np.empty(sample_size, dtype=int)
    if isinstance(init, int):
        X[0] = init
    else:
        cdf0 = np.cumsum(init)
        X[0] = searchsorted(cdf0, u[0])

    # === generate the sample path === #
    for t in range(sample_size-1):
        k = searchsorted(cdfs1d[indptr[X[t]]:indptr[X[t]+1]], u[t+1])
        X[t+1] = indices[indptr[X[t]]+k]

    return X


if __name__ == '__main__':
    P = sparse.csr_matrix([[0.4, 0.6], [0.2, 0.8]])
    init = (0.25, 0.75)
    sample_size = 10
    mc_sample_path_sp(P, init=init, sample_size=sample_size)
