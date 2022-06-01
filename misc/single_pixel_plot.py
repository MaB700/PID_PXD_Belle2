# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
nEventsEach = 800000
removeSinglePixels = False

np.random.seed(123)
slow_pions = pd.read_csv("E:\ML_data/vt/data/slow_pions_evtgen_big.txt",header=None ,comment='#', delimiter= " ", nrows=nEventsEach).values.astype('int')
protons = pd.read_csv("E:\ML_data/vt/data/protons_big.txt",header=None ,comment='#', delimiter= " ", nrows=nEventsEach).values.astype('int')
e_mius = pd.read_csv("E:\ML_data/vt/data/eminus_big.txt",header=None ,comment='#', delimiter= " ", nrows=nEventsEach).values.astype('int')
e_plus = pd.read_csv("E:\ML_data/vt/data/eplus_big.txt",header=None ,comment='#', delimiter= " ", nrows=nEventsEach).values.astype('int')
kaon = pd.read_csv("E:\ML_data/vt/data/kaon_big.txt",header=None ,comment='#', delimiter= " ", nrows=nEventsEach).values.astype('int')
muon = pd.read_csv("E:\ML_data/vt/data/muon_big.txt",header=None ,comment='#', delimiter= " ", nrows=nEventsEach).values.astype('int')
pi_mius = pd.read_csv("E:\ML_data/vt/data/piminus_big.txt",header=None ,comment='#', delimiter= " ", nrows=nEventsEach).values.astype('int')
pi_plus = pd.read_csv("E:\ML_data/vt/data/piplus_big.txt",header=None ,comment='#', delimiter= " ", nrows=nEventsEach).values.astype('int')

# count = np.count_nonzero(hits_train[:, 0] < 0.5)
# print(count)
x = np.arange(0, 256, 1)
y_pion = np.zeros((256,), dtype=int) 
y_proton = np.zeros((256,), dtype=int)
y_e_mius = np.zeros((256,), dtype=int)
y_e_plus = np.zeros((256,), dtype=int)
y_kaon = np.zeros((256,), dtype=int)
y_muon = np.zeros((256,), dtype=int)
y_pi_mius = np.zeros((256,), dtype=int)
y_pi_plus = np.zeros((256,), dtype=int)


for i in range(len(slow_pions)):
    if slow_pions[i, 1] == slow_pions[i, 42]:
        y_pion[slow_pions[i, 1]] += 1

for i in range(len(protons)):
    if protons[i, 1] == protons[i, 42]:
        y_proton[protons[i, 1]] += 1

for i in range(len(e_mius)):
    if e_mius[i, 1] == e_mius[i, 42]:
        y_e_mius[e_mius[i, 1]] += 1

for i in range(len(e_plus)):
    if e_plus[i, 1] == e_plus[i, 42]:
        y_e_plus[e_plus[i, 1]] += 1

for i in range(len(kaon)):
    if kaon[i, 1] == kaon[i, 42]:
        y_kaon[kaon[i, 1]] += 1

for i in range(len(muon)):
    if muon[i, 1] == muon[i, 42]:
        y_muon[muon[i, 1]] += 1

for i in range(len(pi_mius)):
    if pi_mius[i, 1] == pi_mius[i, 42]:
        y_pi_mius[pi_mius[i, 1]] += 1

for i in range(len(pi_plus)):
    if pi_plus[i, 1] == pi_plus[i, 42]:
        y_pi_plus[pi_plus[i, 1]] += 1


y_pion = y_pion / y_pion.sum()
y_proton = y_proton / y_proton.sum()
y_e_mius = y_e_mius / y_e_mius.sum()
y_e_plus = y_e_plus / y_e_plus.sum()
y_kaon = y_kaon / y_kaon.sum()
y_muon = y_muon / y_muon.sum()
y_pi_mius = y_pi_mius / y_pi_mius.sum()
y_pi_plus = y_pi_plus / y_pi_plus.sum()

#plt.figure(figsize=(5, 5))
plt.plot(x, y_pion, '-b', label="slow pion")
plt.plot(x, y_proton, '-g', label="proton")
plt.plot(x, y_e_mius, '-r', label="e-")
plt.plot(x, y_e_plus, '-c', label="e+")
plt.plot(x, y_kaon, '-m', label="kaon")
plt.plot(x, y_muon, '-y', label="muon")
plt.plot(x, y_pi_mius, '-k', label="pi-")
plt.plot(x, y_pi_plus, color='orange', linestyle='solid', label="pi+")

plt.legend(loc="upper right")
plt.xlabel('ADC value')
plt.ylabel('a.u.')
#plt.yscale('log')
#plt.xscale('log')
plt.show()



# %%
