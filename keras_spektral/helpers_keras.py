from operator import mod
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from spektral.data import Dataset, Graph


class MyDataset(Dataset):
    def __init__(self, path, n_samples, label, p=0.75, **kwargs):
        self.data = pd.read_csv(path, header=None, delimiter= " ", nrows=n_samples).values.astype('int')
        self.n_samples = n_samples
        self.label = label
        self.p = p
        super().__init__(**kwargs)

    def read(self):
        def make_graph(index):
            # Load num filled pixels & ADC values
            n = np.count_nonzero(self.data[index, 2:83] > 0)
            adc_indices = np.nonzero(self.data[index, 2:83])
            adc = self.data[index, 2:83][adc_indices]   
            seed_index = np.argmax(adc)      
            y_pos, x_pos = np.divmod(adc_indices, 9)

            # Node features
            x = np.zeros((n, 3)) # num nodes, features
            x[:, 0] = (adc.astype('float32'))/255.0 # ADC value [0,1]
            x[:, 1] = (x_pos.astype('float32'))/8.0 # x_coord [0,1]
            x[:, 2] = (y_pos.astype('float32'))/8.0 # y_coord [0,1]

            # Edges
            a = np.zeros((n, n), dtype=int)
            a[:, seed_index] = 1
            a[seed_index, :] = 1
            a[seed_index, seed_index] = 0 # no self loop
            
            # a = sp.csr_matrix(a) # not needed for small num of nodes
            # dataset creating takes ~ 15x longer when using sp.csr_matrix()

            # Labels
            y = np.array([self.label]).astype('float32')

            # graph features
            # g = np.array([0.5]).astype('float32') #FIXME:

            return Graph(x=x, a=a, y=y)

        # We must return a list of Graph objects
        return [make_graph(i) for i in tqdm(range(self.n_samples))]
