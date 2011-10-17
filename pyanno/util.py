import numpy as np
from scipy.special import gammaln
import time

def random_categorical(distr, nsamples):
    """Return an array of samples from a categorical distribution."""
    assert np.allclose(distr.sum(), 1., atol=1e-8)
    cumulative = distr.cumsum()
    return cumulative.searchsorted(np.random.random(nsamples))


def alloc_vec(N,x=0.0):
    result = []
    n = 0
    while n < N:
        result.append(x)
        n += 1
    return result


def alloc_mat(M,N,x=0.0):
    result = []
    m = 0
    while m < M:
        result.append(alloc_vec(N,x))
        m += 1
    return result


def alloc_tens(M,N,J,x=0.0):
    result = []
    for m in range(M):
        result.append(alloc_mat(N,J,x))
    return result


def alloc_tens4(M,N,J,K,x=0.0):
    result = []
    for m in range(M):
        result.append(alloc_tens(N,J,K,x))
    return result


def fill_vec(xs,y):
    i = 0
    while i < len(xs):
        xs[i] = y
        i += 1


def fill_mat(xs,y):
    i = 0
    while i < len(xs):
        fill_vec(xs[i],y)
        i += 1


def fill_tens(xs,y):
    i = 0
    while i < len(xs):
        fill_mat(xs[i],y)
        i += 1


def vec_copy(x,y):
    n = len(x)
    while (n > 0):
        n -= 1
        y[n] = x[n]


def vec_sum(x):
    sum = 0
    for x_i in x:
        sum += x_i
    return sum


def mat_sum(x):
    sum = 0
    for x_i in x:
        sum += vec_sum(x_i)
    return sum


# TODO this should not be in-place
def prob_norm(theta):
    #raise DeprecationWarning('use normalize instead')
    Z = sum(theta)
    if Z <= 0.0:
        fill_vec(theta,1.0/float(len(theta)))
        return
    n = len(theta) - 1
    while n >= 0:
        theta[n] /= Z
        n -= 1


def normalize(x, dtype=float):
    """Returns a normalized distribution (sums to 1.0)."""
    x = np.asarray(x, dtype=dtype)
    z = x.sum()
    if z <= 0:
        x = np.ones_like(x)
        z = float(len(x))
    return x / z


def create_band_matrix(shape, diagonal_elements):
    diagonal_elements = np.asarray(diagonal_elements)
    def diag(i,j):
        x = np.absolute(i-j)
        x = np.minimum(diagonal_elements.shape[0]-1, x).astype(int)
        return diagonal_elements[x]
    return np.fromfunction(diag, shape)


def warn_missing_vals(varname,xs):
    missing = set(xs) - set(range(max(xs)+1))
    if len(missing) > 0:
        print "Missing values in ",varname,"=",missing


# TODO clean up and simplify and rename
def compute_counts(annotations, nclasses):
    """Transform annotation data in counts format.

    At the moment, it is hard coded for 8 annotators, 3 annotators active at
    any time.

    Input:
    annotations -- Input data (integer array, nitems x 8)
    nclasses -- number of annotation values (# classes)

    Ouput:
    data -- data[i,j] is the number of times the combination of annotators
             number `j` voted according to pattern `i`
             (integer array, nclasses^3 x 8)
    """
    index = np.array([[0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [0, 6, 7],
        [0, 1, 7]], int)
    m = annotations.shape[0]
    n = annotations.shape[1]
    annotations = np.asarray(annotations, dtype=int)

    assert n==8, 'Strange: ' + str(n) + 'annotator number !!!'

    # compute counts of 3-annotator patterns for 8 triplets of annotators
    data = np.zeros((nclasses ** 3, 8), dtype=int)

    # transform each triple of annotations into a code in base `nclasses`
    for i in range(m):
        ind = np.where(annotations[i, :] >= 0)

        code = annotations[i, ind[0][0]] * (nclasses ** 2) +\
               annotations[i, ind[0][1]] * nclasses +\
               annotations[i, ind[0][2]]

        # o is the index of possible combination of annotators in the loop design
        o = -100
        for j in range(8):
            k = 0
            for l in range(3):
                if index[j, l] == ind[0][l]:
                    k += 1
            if k == 3:
                o = j

        if o >= 0:
            data[code, o] += 1
        else:
            print str(code) + " " + str(ind) + " = homeless code"

    return data


def labels_count(annotations, nclasses, missing_val=-1):
    """Compute the total count of labels in observed annotations."""
    valid = annotations!=missing_val
    return np.bincount(annotations[valid], minlength=nclasses)


def labels_frequency(annotations, nclasses, missing_val=-1):
    """Compute the total frequency of labels in observed annotations."""
    valid = annotations!=missing_val
    nobservations = valid.sum()
    return (np.bincount(annotations[valid], minlength=nclasses)
            / float(nobservations))


def string_wrap(st, mode):
    st = str(st)

    if mode == 1:
        st = "\033[1;29m" + st + "\033[0m"
    elif mode == 2:
        st = "\033[1;34m" + st + "\033[0m"
    elif mode == 3:
        st = "\033[1;44m" + st + "\033[0m"
    elif mode == 4:
        st = "\033[1;35m" + st + "\033[0m"
    elif mode == 5:
        st = "\033[1;33;44m" + st + "\033[0m"
    elif mode == 5:
        st = "\033[1;47;34m" + st + "\033[0m"
    else:
        st = st + ' '
    return st


class benchmark(object):
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        print '---- start ----'
        self.start = time.time()
    def __exit__(self,ty,val,tb):
        end = time.time()
        print '---- stop ----'
        print("%s : %0.3f seconds" % (self.name, end-self.start))
        return False


def dirichlet_llhood(theta, alpha):
    """Compute the log likelihood of theta under Dirichlet(alpha)."""
    log_theta = np.log(theta)
    return (gammaln(alpha.sum())
            - (gammaln(alpha)).sum()
            + ((alpha - 1.) * log_theta).sum())
