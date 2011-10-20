import scipy as sp
from pyanno.modelB import ModelB

DATA_PER_LOOP = 100

# TODO replace ModelB with ModelBt and specify target annotator accuracy

# create simple model for 8 annotators
model = ModelB.create_initial_state(nclasses=3, nannotators=8)
# generate data
labels = model.generate_labels(8*DATA_PER_LOOP)
data = model.generate_annotations(labels) + 1
# mask data according to "loop" experimental design
for l in xrange(8):
    label_idx = sp.arange(l+3, l+8) % 8
    data[l*DATA_PER_LOOP:(l+1)*DATA_PER_LOOP, label_idx] = -1

int2str = {1: 'LOW', 2: 'MEDIUM', 3: 'HIGH', -1: 'NA'}
anno = []
for i in range(data.shape[0]):
    anno.append([])
    for j in range(data.shape[1]):
        anno[-1].append(int2str[data[i,j]])

# save data
sp.savetxt('testdata_words.txt', anno, fmt='%s')
