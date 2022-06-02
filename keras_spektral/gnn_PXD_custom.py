# %%
""" Import librarys """
import numpy as np
import sklearn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.layers import GCSConv, GlobalAvgPool, GCNConv, ARMAConv
from spektral.transforms.normalize_adj import NormalizeAdj

import wandb

from helpers_keras import *

# %%
""" Create graphs """
batch_size = 1024
epochs = 30
es_patience = 5
nEventsEach = 50000

data = MyDataset(path="E:\ML_data/vt/data/slow_pions_evtgen_big.txt", label = 1.0, n_samples=nEventsEach, transforms=NormalizeAdj()) \
     + MyDataset(path="E:\ML_data/vt/data/protons_big.txt", label = 0.0, n_samples=nEventsEach, transforms=NormalizeAdj())
#data.filter(lambda g: g.n_nodes > 2)
# %%
# Train/valid/test split
np.random.seed(123)
idxs = np.random.permutation(len(data))
idx_train, idx_val, idx_test = np.split(idxs, [int(0.6 * len(data)), int(0.8 * len(data))])

# Data loaders
loader_train = DisjointLoader(data[idx_train], batch_size=batch_size, epochs=epochs)
loader_val = DisjointLoader(data[idx_val], batch_size=batch_size)
loader_test = DisjointLoader(data[idx_test], batch_size=batch_size)
# del data

# %%
""" Create model """
node_features = data[0].n_node_features
S = data[0].n_edge_features
x_in = Input(shape=(node_features, ), name='node_features')
a_in = Input(shape=(None, ), name='adjacency_matrix')
# e_in = Input(shape=(S, ), name='edge_features') # 
i_in = Input(shape=(), dtype=tf.int64, name='Graph_IDs')

x = ARMAConv(32, activation='relu')([x_in, a_in])
x = ARMAConv(32, activation='relu')([x, a_in])
x = ARMAConv(32, activation='relu')([x, a_in])
out = GlobalAvgPool()([x, i_in])
out = Dense(1, activation="sigmoid")(out)

model = Model(inputs=[x_in, a_in, i_in], outputs=out)

optimizer = Adam(learning_rate=1e-3)
loss_fn = BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss_fn)
model.summary()

# %%
""" Train """

wandb.init(project="PXD_test")
config = wandb.config
#wandb.watch(model)

@tf.function(input_signature=loader_train.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(binary_accuracy(target, predictions))
    return loss, acc

def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(binary_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

epoch = step = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience
results = []
for batch in loader_train:
    step += 1
    loss, acc = train_step(*batch)
    results.append((loss, acc))
    if step == loader_train.steps_per_epoch:
        step = 0
        epoch += 1

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_val)
        train_loss, train_acc = np.mean(results, 0)
        print(
            "Ep. {} - Loss: {:.4f} - Acc: {:.4f} - Val loss: {:.4f} - Val acc: {:.4f}".format(
                epoch, train_loss, train_acc, val_loss, val_acc
            )
        )

        wandb.log({ "train_loss": train_loss,
                    "train_accuracy": train_acc, 
                    "val_loss": val_loss,
                    "val_accuracy": val_acc
                 })

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            print("New best val_loss {:.4f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
        results = []

# %%
""" Evaluate on test set """
model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(loader_test)
print("Test loss: {:.4f}. Test acc: {:.4f}".format(test_loss, test_acc))

def predict(loader):
    tar = np.empty((0))
    prd = np.empty((0))
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        tar = np.append(tar, target)
        prd = np.append(prd, np.array(pred))
    return tar, prd

test_gt, test_pred = predict(loader_test)
# %%
test_auc = roc_auc_score(test_gt, test_pred)
print("Test AUC: {:.4f}".format(test_auc))

wandb.log({"test_loss": test_loss,
           "test_acc": test_acc,
           "test_auc": test_auc})

wandb.log({"roc": wandb.plot.roc_curve( test_gt, 
                                        np.concatenate(((1-test_pred).reshape(-1,1),test_pred.reshape(-1,1)),axis=1), 
                                        classes_to_plot=[1])})

wandb.log({"cm": wandb.plot.confusion_matrix(   probs=None, 
                                                y_true=[1 if a_ > 0.5 else 0 for a_ in test_gt], 
                                                preds=[1 if a_ > 0.5 else 0 for a_ in test_pred], 
                                                class_names=["background", "signal"])})
