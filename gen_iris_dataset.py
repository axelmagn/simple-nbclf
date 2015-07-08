import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split

def dump(arr, fname):
    with open(fname, 'w') as f:
        f.write('\n'.join(['\t'.join( [str(cell) for cell in line]) for line in arr]))

data = load_iris()
X = data.data.astype(np.int32)
y = label_binarize(data.target, [0,1,2])
labels = data.target_names
data = {}
data['x_train'], data['x_test'], data['y_train'], data['y_test'] = train_test_split(X, y, test_size=0.2)
for fname in ['x_train', 'x_test', 'y_train', 'y_test']:
    dump(data[fname], "data/iris_%s.tsv" % fname)

