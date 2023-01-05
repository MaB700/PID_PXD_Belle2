import numpy as np
import pandas as pd
import time
import uproot

n_samples = 1_000_000
path = "E:/ML_data/vt/data/slow_pions_evtgen_big.txt"
start = time.time()
data = pd.read_csv(path, header=None, delimiter= " ", nrows=n_samples).values.astype('float32')[:, 2:83]
print("Time to read csv in ms: ", time.time() - start)

start = time.time()
x, g = None, None
with uproot.open("E:/ML_data/vt/data/data.root") as file:
    x = np.array(file["data"]["adc"].array(entry_stop=n_samples))
    g = np.array(file["data"]["pos"].array(entry_stop=n_samples))

print("Time to read root in ms: ", time.time() - start)
print(data[0])
print(x[0])
# check if data and x is the same
np.testing.assert_allclose(data, x, atol=1e-7, rtol=1e-5, verbose=True)