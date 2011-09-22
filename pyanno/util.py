import scipy as sp


def random_categorical(distr, nsamples):
    """Return an array of samples from a categorical distribution."""
    assert sp.allclose(distr.sum(), 1., atol=1e-8)
    cumulative = distr.cumsum()
    return cumulative.searchsorted(sp.random.random(nsamples))


def log_beta_pdf(x, a, b):
    """Return the natural logarithm of the Beta(a,b) distribution at x."""
    log_gamma = sp.special.gammaln
    return (log_gamma(a+b) - log_gamma(a) - log_gamma(b)
            + (a-1.)*sp.log(x) + (b-1.)*sp.log(1.-x))


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
    x = sp.asarray(x, dtype=dtype)
    z = x.sum()
    if z <= 0:
        x = sp.ones_like(x)
    return x / x.sum()


def create_band_matrix(shape, diagonal_elements):
    diagonal_elements = sp.asarray(diagonal_elements)
    def diag(i,j):
        x = sp.absolute(i-j)
        x = sp.minimum(diagonal_elements.shape[0]-1, x).astype(int)
        return diagonal_elements[x]
    return sp.fromfunction(diag, shape)


def warn_missing_vals(varname,xs):
    missing = set(xs) - set(range(max(xs)+1))
    if len(missing) > 0:
        print "Missing values in ",varname,"=",missing
