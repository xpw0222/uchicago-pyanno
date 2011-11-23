
def setup_package():
    """Setup function for testing with nosetests.
    """

    # remove noise coming from numpy regarding log(0) and log(-inf),
    # those cases are already taken care of in the tests
    import numpy
    numpy.seterr(divide='ignore', invalid='ignore')
