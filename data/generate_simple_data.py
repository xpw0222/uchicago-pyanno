import scipy as sp
from pyanno.modelB import ModelB

DATA_PER_LOOP = 100

# TODO replace ModelB with ModelBt and specify target annotator accuracy

# create simple model for 8 annotators
model = ModelB.create_initial_state(nclasses=4, nannotators=8, nitems=8*DATA_PER_LOOP)
# generate data
labels = model.generate_labels()
data = model.generate_annotations(labels).T + 1
# mask data according to "loop" experimental design
for l in xrange(8):
    label_idx = sp.arange(l+3, l+8) % 8
    data[l*DATA_PER_LOOP:(l+1)*DATA_PER_LOOP, label_idx] = -1
# save data
sp.savetxt('testdata.txt', data, fmt='%d')
