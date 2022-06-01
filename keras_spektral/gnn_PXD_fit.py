# %%
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.transforms.normalize_adj import NormalizeAdj


tf.keras.backend.set_floatx('float32')

# %%
class MyDataset(Dataset):
   
    def __init__(self, path, n_samples, label, p=0.75, **kwargs):
        self.data = pd.read_csv(path, header=None, delimiter= " ", nrows=n_samples).values.astype('int')
        self.n_samples = n_samples
        self.label = label # astype float32 ?
        self.p = p
        self.index = 0
        super().__init__(**kwargs)

    def read(self):
        def make_graph():
            # Load num filled pixels & ADC values
            n = np.count_nonzero(self.data[self.index, 2:83] > 0)
            adc_indices = np.nonzero(self.data[self.index, 2:83])
            adc = self.data[self.index, 2:83][adc_indices]
            
            # Node features
            x = np.zeros((n, 1)) # num features = 1
            x[:, 0] = (adc.astype('float32'))/255.0
            # x[:, 1] = feature 2

            # Edges
            a = np.random.rand(n, n) <= self.p
            a = np.maximum(a, a.T).astype(int)
            # a = sp.csr_matrix(a) # not needed for small num of nodes
            # dataset creating takes ~ 15x longer when using sp.csr_matrix()

            # Labels
            y = np.array([self.label]).astype('float32')
            
            self.index += 1
            
            return Graph(x=x, a=a, y=y)

        # We must return a list of Graph objects
        return [make_graph() for _ in range(self.n_samples)]

# %%
nEventsEach = 200000
pion_graphs = MyDataset(path="E:\ML_data/vt/data/slow_pions_evtgen_big.txt", label = 1.0, n_samples=nEventsEach, transforms=NormalizeAdj())
bg_graphs = MyDataset(path="E:\ML_data/vt/data/protons_big.txt", label = 0.0, n_samples=nEventsEach, transforms=NormalizeAdj())
data = pion_graphs + bg_graphs
#data.filter(lambda g: g.n_nodes > 3)
print(data)

# %%
batch_size = 512
epochs = 20
es_patience = 20
# Train/valid/test split
np.random.seed(123)
idxs = np.random.permutation(len(data))
split_va, split_te = int(0.8 * len(data)), int(0.9 * len(data))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
data_tr = data[idx_tr]
data_va = data[idx_va]
data_te = data[idx_te]

# Data loaders
loader_tr = DisjointLoader(data_tr, batch_size=32, epochs=epochs)
loader_va = DisjointLoader(data_va, batch_size=32)
loader_te = DisjointLoader(data_te, batch_size=32)

# %%
class Net(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = GCSConv(32, activation="relu")
        self.conv2 = GCSConv(32, activation="relu")
        self.conv3 = GCSConv(32, activation="relu")
        self.global_pool = GlobalAvgPool()
        self.dense = Dense(data.n_labels, activation="sigmoid")

    def call(self, inputs):
        x, a, i = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        output = self.global_pool([x, i])
        output = self.dense(output)

        return output

model = Net()
lr = 1e-3
optimizer = Adam(learning_rate=lr)
loss_fn = BinaryCrossentropy()
model.compile(optimizer, loss_fn , metrics=['acc'])

# %%
model.fit(
        loader_tr.load(), 
        steps_per_epoch=loader_tr.steps_per_epoch, 
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        epochs=10
                )

# %%
def evaluate_mean_error(loader):
    output = 0
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        output += np.average(np.absolute(np.array(pred) - target))
        
    return output/loader.steps_per_epoch

a = evaluate_mean_error(loader_va)
print(a)

# %%
from sklearn.metrics import roc_auc_score

def evaluatex(loader):
    tar = np.empty((0))
    prd = np.empty((0))
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        #output += np.average(np.absolute(np.array(pred) - target))
        tar = np.append(tar, target)
        prd = np.append(prd, np.array(pred))
    return prd, tar

p, t = evaluatex(loader_te)
auc = roc_auc_score(t, p)
print(auc)


