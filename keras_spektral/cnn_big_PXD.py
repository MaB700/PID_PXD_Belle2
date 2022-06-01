# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model


print('Tensorflow version: ' + tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT available")

# %%
nEventsEach = 50000
removeSinglePixels = False

np.random.seed(123)
pions = pd.read_csv("E:\ML_data/vt/data/slow_pions_evtgen_big.txt",header=None ,comment='#', delimiter= " ", nrows=nEventsEach).values.astype('int')
protons = pd.read_csv("E:\ML_data/vt/data/protons_big.txt",header=None ,comment='#', delimiter= " ", nrows=nEventsEach).values.astype('int')
protons[:, 0] = 0
# count = np.count_nonzero(hits_train[:, 0] < 0.5)
# print(count)
if removeSinglePixels :
    pions = pions[ pions[:, 1] != pions[:, 42] ]
    protons = protons[ protons[:, 1] != protons[:, 42] ]


conc = np.concatenate([pions, protons])
np.random.shuffle(conc)
input = np.reshape((conc[:, 2:83].astype('float32'))/255.0, (-1, 9, 9, 1))
tar = (np.reshape(conc[:, 0], (-1, 1))).astype('float32')

# %%
# x = np.arange(0, 256, 1)
# y_pion = np.ones((256,), dtype=int) 
# y_proton = np.ones((256,), dtype=int)

# for i in range(len(pions)):
#     if pions[i, 1] == pions[i, 42]:
#         y_pion[pions[i, 1]] += 1

# for i in range(len(protons)):
#     if protons[i, 1] == protons[i, 42]:
#         y_proton[protons[i, 1]] += 1

# plt.plot(x, y_pion, 'b--', x, y_proton, 'r--')
# plt.yscale('log')
# plt.show()
# print(y_pion.sum())
# print(y_proton.sum())

# %%
node_num = 3*81
model = Sequential()
model.add(InputLayer(input_shape=(9, 9, 1)))
# model.add(Conv2D(filters=15, kernel_size=3, padding='same', activation='relu'))
# model.add(Conv2D(filters=15, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=15, kernel_size=3, padding='same', activation='relu'))

model.add(Flatten())
model.add(Dense(units=node_num, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=node_num, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=node_num, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=node_num, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=node_num, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(units=8, activation='relu'))
# model.add(Dense(units=32, activation='relu'))
# model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='min')
model.compile(optimizer=opt, loss='bce', metrics=['acc', tf.keras.metrics.BinaryAccuracy()])#tf.keras.metrics.BinaryAccuracy() , compare both

model.fit(  x=input, y=tar,
            validation_split=0.2,
            batch_size=1024,
            epochs=100,
            shuffle=True,
            callbacks=[es])

# %%
pred = model.predict(input, batch_size = 1024)
predx = np.abs(tar - pred)
predxx = predx[  predx[:, 0] > 0.1 ]


