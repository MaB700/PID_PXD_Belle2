# %%
import numpy as np
import pandas as pd
import csv

# %%
def count_num_pixels(path):
    data = pd.read_csv(path, header=None, delimiter= " ").values
    
    def count(index):
        return np.count_nonzero(data[index, 2:83] > 0)
    
    pixels = [0, 0, 0, 0]

    for i in range(len(data)):
        n = count(i)
        pixels[0] += 1
        if n == 1 :
            pixels[1] += 1
        elif n == 2 :
            pixels[2] += 1
        else :
            pixels[3] += 1
    
    print(pixels)

def create_balanced_data(paths, amount_per_class = 5):
    np_arr_list = []
    for path in paths:
        datax = pd.read_csv(path, header=None, delimiter= " ").values
        np_arr_list.append(datax)
    
    data = np.vstack(np_arr_list)
    np.random.shuffle(data)

    n1 = 0
    n2 = 0
    n3p = 0
    out = np.zeros((3*amount_per_class, 86))
    j = 0
    max_index = 0
    for i in range(len(data)):
        max_index = i + 1
        n = np.count_nonzero(data[i, 2:83] > 0)
        if n == 1 and n1 < amount_per_class:
            out[j] = data[i]
            j += 1
            n1 += 1
        elif n == 2 and n2 < amount_per_class:
            out[j] = data[i]
            j += 1
            n2 += 1
        elif n >= 3 and n3p < amount_per_class:
            out[j] = data[i]
            j += 1
            n3p += 1
        elif n1 == n2 == n3p == amount_per_class :
            break
    
    np.random.shuffle(out)    
    print(len(data))
    print(max_index)

    with open("data_balanced_train.txt", "w", newline='') as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(out)
    
    with open("data_train.txt", "w", newline='') as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(data[0:3*amount_per_class])

    with open("data_test.txt", "w", newline='') as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(data[max_index: max_index+3*amount_per_class])    

path0 = "E:\ML_data/vt/data/slow_pions_evtgen_big.txt"
path1 = "E:\ML_data/vt/data/pions_big.txt"
path2 = "E:\ML_data/vt/data/protons_big.txt"
path3 = "E:\ML_data/vt/data/muon_big.txt"
path4 = "E:\ML_data/vt/data/electron_big.txt"
path5 = "E:\ML_data/vt/data/kaon_big.txt"
path6 = "E:\ML_data/vt/data/gamma_big.txt"
path7 = "E:\ML_data/vt/data/piplus_big.txt"
path8 = "E:\ML_data/vt/data/piminus_big.txt"
path9 = "E:\ML_data/vt/data/eminus_big.txt"
path10 = "E:\ML_data/vt/data/eplus_big.txt"

create_balanced_data([path0], 300000)
#create_balanced_data([path1, path2, path3, path4, path5, path6, path7, path8, path9, path10], 300000)
count_num_pixels("data_balanced_train.txt")
count_num_pixels("data_train.txt")
count_num_pixels("data_test.txt")
