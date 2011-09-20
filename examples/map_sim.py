import numpy as np
try:
    import pyanno.multinom
    import pyanno.util
    from pyanno.util import normalize, create_band_matrix
    from pyanno.models import ModelB
except ImportError, e:
    print e
    print ""
    print "Need to install pyanno package and dependencies."
    print "See instructions in Install.txt from pyanno distribution"
    raise SystemExit(1)

# Simulated Sizes
nitems = 200
nannotators = 5
nclasses = 4

print "SIZES"
print "# items=",nitems
print "# annotators=",nannotators
print "# classes=",nclasses


print "SIMULATING ORDINAL CODING DATA SET"
# create random model (this is our graund truth model)
true_model = ModelB.random_model(nclasses, nannotators, nitems)
# create random data
labels = true_model.generate_labels()
annotations = true_model.generate_annotations(labels)


print "CALCULATING SAMPLE PARAMETERS"
# create a new model, initialize w. sample parameters

sample_pi = normalize(np.bincount(labels))

sample_theta = np.zeros((nannotators, nclasses, nclasses))
for j in xrange(nannotators):
    for k in xrange(nclasses):
        label_is_k = annotations[j, labels==k]
        if len(label_is_k) > 0:
            count = np.bincount(label_is_k, minlength=nclasses)
            sample_theta[j,k,:] = normalize(count)
        else:
            sample_theta[j,k,:] = np.ones((nclasses,)) / nclasses


alpha = create_band_matrix((nclasses, nclasses), [4., 2., 1.])
beta = np.ones(nclasses)

model = ModelB(nclasses, nannotators, nitems, sample_pi, sample_theta,
               alpha, beta)

print "PRIORS"
print "    alpha=",alpha
print "    beta=",beta


print "RUNNING EM"

epsilon = 0.00001
init_acc = 0.6
(diff,ll,lp,cat_map) = model.map(annotations, epsilon, init_acc)

print "CONVERGENCE ll[final] - ll[final-10]=",diff


print "PREVALENCE ESTIMATES"
print "{0:>2s}, {1:>5s}, {2:>5s}, {3:>5s}, {4:>6s}, {5:>6s}".\
        format("k","sim","samp","MAP","d.sim","d.samp")
prev = true_model.pi
prevalence_sample = sample_pi
prev_map = model.pi
for k in xrange(nclasses):
    print "{0:2d}, {1:5.3f}, {2:5.3f}, {3:5.3f}, {4:+5.3f}, {5:+5.3f}".\
            format(k,prev[k],prevalence_sample[k],prev_map[k],
                   prev_map[k]-prev[k],
                   prev_map[k]-prevalence_sample[k])

print "ACCURACY ESTIMATES"
print "{0:>3s},{1:>2s},{2:>2s}, {3:>5s}, {4:>5s}, {5:>5s}, {6:>6s}, {7:>6s}".\
        format("j","k1","k2","sim","samp","map","d.sim","d.samp")
accuracy = true_model.theta
sample_theta = sample_theta
accuracy_map = model.theta
for j in xrange(nannotators):
    for k1 in xrange(nclasses):
        for k2 in xrange(nclasses):
            print "{0:3d},{1:2d},{2:2d}, {3:5.3f}, {4:5.3f}, {5:5.3f}, {6:+5.3f}, {7:+5.3f}".\
                    format(j,k1,k2,accuracy[j][k1][k2],
                           sample_theta[j][k1][k2],
                           accuracy_map[j][k1][k2],
                           accuracy_map[j][k1][k2]-accuracy[j][k1][k2],
                           accuracy_map[j][k1][k2]-sample_theta[j][k1][k2])

print "CATEGORY ESTIMATES"
for i in xrange(nitems):
    print "{0:5d}".format(i),
    for k in xrange(nclasses):
        match = "*" if labels[i] == k else " "
        print " {0:2d}:{1:1}{2:5.3f}".format(k,match,cat_map[i][k]),
    print ""
