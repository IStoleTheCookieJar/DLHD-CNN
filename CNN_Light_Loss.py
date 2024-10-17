#!/usr/bin/env python
# coding: utf-8

# Imports
import time
start_time = time.time()
import json
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LinearRegression

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import optuna

import scipy

import sys
import glob
from copy import deepcopy
soft = 1e-7

# Need 3 system arguments:
# 1st argument: trial number
# 2nd argument: date of trial in ("mm_dd_yyyy")
# 3rd argument: number of channels for this run: "1" , "2" , "3," ...
# 4th argument: Whether it is TNG300 or TNG100 {"300","100"}
# 5th set of undefined arguments: type of data/data names: 
    # "0.2-0.4_lem"
    # "0.4-0.6_lem"
    # "0.73-1.1_lem"
    # "1.43-2.0_lem"
    # "0.2-0.4_ccd"
    # "0.4-0.6_ccd"
    # "0.7-1.1_ccd"
    # "1.4-2.0_ccd"
    # "Fe_XVII"
    # "O_VII"
    # "O_VIII"
    # "H1"
    # "H1_vel"
    # "H1_disp"
# 6th set of undefined arguments: parameters to test on:
    # TNG300: "ID","M500c","f_gas_R500c","a_form","conc_200c","shape","Gamma_R500C"\
             # ,"Gamma_vir","Log_T","Log_Z","O/Fe","Ne/Fe","C/O","axis placeholder"\
             # ,"M500c","M200c","f_gas_0.5R200c","f_gas_0.7R200c","f_gas_R200c","f_gas_2R200c"
    # TNG100: "ID","M200c","f_gas_R500c","a_form","conc_200c","shape","f_gas_R200C"\
             # ,"Log_M_star_","Log_T_","Log_Z_","O/Fe","Ne/Fe","C/O","axis placeholder"\
             # ,"M500c","M200c","f_gas_0.5R200c","f_gas_0.7R200c","f_gas_R200c","f_gas_2R200c"

# # Class definition
# In[2]:

if(len(sys.argv) < 6):
    sys.exit("Not enough arguments, check CNN_Light.py for argument requirements")

if(sys.argv[1] != ""):
    trial_number = sys.argv[1]
else:
    sys.exit("No valid argument provided for trial_number")
    
if(sys.argv[2] != ""):
    date = sys.argv[2]
else:
    sys.exit("No valid argument provided for data labels")

    
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
if(torch.cuda.is_available()):
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")

    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

#--------------------
# Checking Channel and parameter inputs
#--------------------
### Channels
check_numerical = sys.argv[3].isnumeric()
if(check_numerical == False):
    sys.exit("No valid argument provided for channel size, try: 1 , 2 , 3, ...")
else:
    num_channels = int(sys.argv[3])

### Parameters
check_param_numerical = sys.argv[5+num_channels].isnumeric()
if(check_param_numerical == False):
    sys.exit("No valid argument provided for number of parameters, try 1, 2, 3, ...")
else:
    num_params = int(sys.argv[5+num_channels])

#---------------------
#LOAD DATA
#---------------------

# TNG300
Channel_300_keys = {"0.2-0.4_ccd":"/projects/beop5934/halos/Groups/Maps_0.2_0.4keV_ccd_TNG300_Groups_z=0.00.npy",\
                    "0.4-0.6_ccd":"/projects/beop5934/halos/Groups/Maps_0.4_0.6keV_ccd_TNG300_Groups_z=0.00.npy",\
                    "0.7-1.1_ccd":"/projects/beop5934/halos/Groups/Maps_0.7_1.1keV_ccd_TNG300_Groups_z=0.00.npy",\
                    "1.4-2.0_ccd":"/projects/beop5934/halos/Groups/Maps_1.4_2.0keV_ccd_TNG300_Groups_z=0.00.npy",\
                    "0.2-0.4_lem":"/projects/beop5934/halos/Groups/Maps_0.2_0.4keV_lem_TNG300_Groups_z=0.00.npy",\
                    "0.4-0.6_lem":"/projects/beop5934/halos/Groups/Maps_0.4_0.6keV_lem_TNG300_Groups_z=0.00.npy",\
                    "0.73-1.1_lem":"/projects/beop5934/halos/Groups/Maps_0.73_1.1keV_lem_TNG300_Groups_z=0.00.npy",\
                    "1.43-2.0_lem":"/projects/beop5934/halos/Groups/Maps_1.43_2.0keV_lem_TNG300_Groups_z=0.00.npy",\
                    "Fe_XVII":"/projects/beop5934/halos/Groups/Maps_Fe_XVII_TNG300_Groups_z=0.00.npy",\
                    "O_VII":"/projects/beop5934/halos/Groups/Maps_O_VII_TNG300_Groups_z=0.00.npy",\
                    "O_VIII":"/projects/beop5934/halos/Groups/Maps_O_VIII_TNG300_Groups_z=0.00.npy",\
                    "H1":"/pl/active/CASA/beop5934/halos/TNG300/Maps_H_I_coldens_TNG300_z=0.00.npy",\
                    "H1_vel":"/pl/active/CASA/beop5934/halos/TNG300/Maps_H_I_vel_TNG300_z=0.00.npy",\
                    "H1_disp":"/pl/active/CASA/beop5934/halos/TNG300/Maps_H_I_disp_TNG300_z=0.00.npy"}
            

#TNG100

Channel_100_keys = {"0.2-0.4_lem":"/pl/active/CASA/kaad8904/Maps_0.2_0.4keV_lem_TNG100_z=0.010.1000ksec.npy",\
                    "0.4-0.6_lem":"/pl/active/CASA/kaad8904/Maps_0.4_0.6keV_lem_TNG100_z=0.010.1000ksec.npy",\
                    "0.73-1.1_lem":"/pl/active/CASA/kaad8904/Maps_0.73_1.1keV_lem_TNG100_z=0.010.1000ksec.npy",\
                    "1.43-2.0_lem":"/pl/active/CASA/kaad8904/Maps_1.43_2.0keV_lem_TNG100_z=0.010.1000ksec.npy",\
                    "0.2-0.4_ccd":"/pl/active/CASA/beop5934/halos/TNG100/Maps_0.2_0.4keV_ccd_TNG100_z=0.010.1000ksec.npy",\
                    "0.4-0.6_ccd":"/pl/active/CASA/beop5934/halos/TNG100/Maps_0.4_0.6keV_ccd_TNG100_z=0.010.1000ksec.npy",\
                    "0.7-1.1_ccd":"/pl/active/CASA/beop5934/halos/TNG100/Maps_0.7_1.1keV_ccd_TNG100_z=0.010.1000ksec.npy",\
                    "1.4-2.0_ccd":"/pl/active/CASA/beop5934/halos/TNG100/Maps_1.4_2.0keV_ccd_TNG100_z=0.010.1000ksec.npy",\
                    "Fe_XVII":"/pl/active/CASA/kaad8904/Maps_Fe_XVII_TNG100_z=0.010.1000ksec.npy",\
                    "O_VII":"/pl/active/CASA/kaad8904/Maps_O_VII_TNG100_z=0.010.1000ksec.npy",\
                    "O_VIII":"/pl/active/CASA/kaad8904/Maps_O_VIII_TNG100_z=0.010.1000ksec.npy",\
                    "H1":"/pl/active/CASA/beop5934/halos/TNG100/Maps_H_I_coldens_TNG100_z=0.000.npy",\
                    "H1_vel":"/pl/active/CASA/beop5934/halos/TNG100/Maps_H_I_vel_TNG100_z=0.000.npy",\
                    "H1_disp":"/pl/active/CASA/beop5934/halos/TNG100/Maps_H_I_disp_TNG100_z=0.000.npy"}

# Accessing Truth values
if(sys.argv[4] == "300"):
    Names = ["ID","M500c","f_gas_R500c","a_form","conc_200c","shape","Gamma_R500C"\
             ,"Gamma_vir","Log_T","Log_Z","OFe","NeFe","CO","axis placeholder"\
             ,"M500c_","M200c","f_gas_0.5R200c","f_gas_0.7R200c","f_gas_R200c","f_gas_2R200c"]
    Dataset_overlord = pd.read_csv("/pl/active/CASA/beop5934/halos/TNG300/Params_Groups_TNG300_z=0.00.ascii_ID_fgas"\
                                   , delimiter = "\s+",names = Names)
    DO_new = pd.DataFrame(np.repeat(Dataset_overlord.values,3,axis=0))
    DO_new.columns = Dataset_overlord.columns
    # Dataset_overlord = Dataset_overlord.append((Dataset_overlord,Dataset_overlord))
    # Dataset_overlord = Dataset_overlord.sort_values('Id',ignore_index=True)
    # TM500 = Dataset_overlord['T:Halo_Mass']
    # F_g_500 = Dataset_overlord["T:F_gas"]
    # Mass_cut = TM500 >= 13.2 # cutting too low mass items
    # Dataset_overlord = Dataset_overlord[Mass_cut].reset_index() # cutting
    # ids = Dataset_overlord['Id'] # cutting
    # TM500 = Dataset_overlord['T:Halo_Mass'] # cutting
    # print(len(TM500)) # cutting length
    
elif(sys.argv[4] == "100"):
    Names_100 = ["ID","M200c","f_gas_R500c", "a_form","conc_200c","shape","Log_M_star","M500c",
                 "f_gas_0.5R200c","f_gas_0.7R200c","f_gas_R200c","f_gas_2R200c",
                 "Log_Z_50","Log_Z_100","Log_Z_200","OFe_50","OFe_100","OFe_200","f_cool_R500c"]
    Dataset_overlord = pd.read_csv("/pl/active/CASA/beop5934/halos/TNG100/Params_TNG100_z=0.00.multiradii.ascii_ID_fgas"\
                                       , delimiter = "\s+",names = Names_100)
    DO_new = pd.DataFrame(np.repeat(Dataset_overlord.values,3,axis=0))
    DO_new.columns = Dataset_overlord.columns
    # Dataset_overlord = Dataset_overlord.sort_values('Id',ignore_index=True)
    # # Dataset_overlord_100["Id"] = Dataset_overlord_100.index
    # TM200 = Dataset_overlord_100['T:Halo_Mass'] # TNG M200 mass
    # F_g_200 = Dataset_overlord_100["T:F_gas"]
    # Mass_cut = TM500 >= 13.2 # cutting too low mass items
    # Dataset_overlord = Dataset_overlord[Mass_cut].reset_index() # cutting
    # ids = Dataset_overlord['Id'] # cutting
    # TM500 = Dataset_overlord['T:Halo_Mass'] # cutting
    # print(len(TM500)) # cutting length
    
else:
    sys.exit("Invalid Argument for Simulation (arg 4) try 300 or 100")

if(sys.argv[4] == "300"):    
    for i in range(5,5+num_channels):
        for j in Channel_300_keys:
            if(sys.argv[i] == j):
                if(i-5 == 0):
                    temp = np.load(Channel_300_keys[j])
                    shape = list(temp.shape)
                    shape.insert(0,num_channels)
                    shape = tuple(shape)
                    images = np.zeros(shape)
                img = np.load(Channel_300_keys[j])
                if(sys.argv[i] == "H1_vel" or sys.argv[i] == 'H1_disp'):
                    img = img
                else:
                    img = np.log10(img)
                print(f"Channel = {j},Mean of channel = {np.mean(img):.3f}, std of channel = {np.std(img):.3f}")
                img = (img-np.mean(img))/np.std(img)
                print(f"Range: {np.min(img):.3f} - {np.max(img):.3f}")
                images[i-5] = img
                print(sys.argv[i])
elif(sys.argv[4] == "100"):
    for i in range(5,5+num_channels):
        for j in Channel_100_keys:
            if(sys.argv[i] == j):
                if(i-5 == 0):
                    temp = np.load(Channel_100_keys[j])
                    shape = list(temp.shape)
                    shape.insert(0,num_channels)
                    shape = tuple(shape)
                    images = np.zeros(shape)
                img = np.load(Channel_100_keys[j])
                if(sys.argv[i] == "H1_vel" or sys.argv[i] == 'H1_disp'):
                    img = img
                else:
                    img = np.log10(img)
                print(f"Channel = {j},Mean of channel = {np.mean(img):.3f}, std of channel = {np.std(img):.3f}")
                img = (img-np.mean(img))/np.std(img)
                print(f"Range: {np.min(img):.3f} - {np.max(img):.3f}")
                images[i-5] = img
                print(sys.argv[i])

#--------------------
# Gathering IDs
#--------------------
ids = DO_new['ID']
ids_new = []
for i in range(0,len(ids),3):
    ids_new.append(f"{int(ids[i])}_x")
    ids_new.append(f"{int(ids[i])}_y")
    ids_new.append(f"{int(ids[i])}_z")
ids = ids_new
    
    
#-----------------
# Setting Parameter Array
#-----------------
param_titles = []
params = np.zeros((len(ids),num_params))
for i in range(5+num_channels+1,5+num_channels+1+num_params): 
    # 5 is where the chanels start
    # +num_channels is where num_params is defined
    # +1 is where the start of param inputs is
    # +num_params is where the param inputs should end
    try:
        param_temp = DO_new[sys.argv[i]]
    except KeyError:
        sys.exit("Param ",i," does not exist, try again")
    except:
        sys.exit("A different error from KeyError has occured in finding the parameters")
    param_titles.append(sys.argv[i])
    params[:,i-5-num_channels-1] = param_temp.values
print(param_titles)
print(params)
    
#-------------------------------------
### Training and Testing Dataset setup
#-------------------------------------

rand_start = np.arange(images.shape[1])
print('Starting array = ',rand_start)
np.random.seed(125431)
# torch.manual_seed(571753)
np.random.shuffle(rand_start)
print('Shuffled array = ',rand_start)



split_train = int(images.shape[1]*0.8)
split_test = images.shape[1]-split_train


print(f"training split looks like {split_train} galaxies \ntesting split looks like {split_test} galaxies")
train_indeces = rand_start[:split_train]
test_indeces = rand_start[split_train:]
print("Training indeces = ",train_indeces,"\nTesting indeces = ",test_indeces)
print(f"Shape of train = {train_indeces.shape}\nShape of test = {test_indeces.shape}")
print("Shape of Data ",images.shape)


#--------------------
# Gathering Truth Values
#--------------------
merged_x = {}
for i in range(images.shape[1]):
    merged_x[ids[i]] = torch.tensor(
        images[:,i],
        dtype=torch.float32
    )
merged_y = dict( # flattened
    zip(
        ids,
        torch.tensor(params,dtype=torch.float32)
    )
)
print(len(merged_x))
print(len(merged_y))


class CNN_Clusters(Dataset):
    def __init__(self,x,y,clusters_id):
        self.cluster_ids = clusters_id
        self.total_samples = len(self.cluster_ids)
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        cluster_id = self.cluster_ids[idx]
        x = self.x[cluster_id]
        y = self.y[cluster_id]
        return x, y, x.shape[0], cluster_id

def collate_fn(batch):
    x, y, sizes, batch_ids = zip(*batch)
#     x_batched = torch.concat(x) #1D CNN
    x_batched = torch.stack(x)
    y_batched = torch.stack(y)
    return x_batched, y_batched, sizes, batch_ids


clid_train = [] # flattened string cluster ids
clid_test = []
for i in train_indeces: clid_train.append(ids[i])
for i in test_indeces: clid_test.append(ids[i])
print(clid_train[0])
print(len(clid_train))
print(clid_test[0])
print(len(clid_test))



def build_loader(clids, **kwargs):
    dataset = CNN_Clusters(merged_x, merged_y,clids)
    loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn,
                        **kwargs)
    return loader

train_loader = build_loader(clid_train, shuffle=False)

test_loader = build_loader(clid_test, shuffle=False)

all_loader = build_loader(ids, shuffle = False)
print(train_loader.batch_size)
print(train_loader.dataset[0][0].shape)
print(all_loader.batch_size)

#-----------------
#DEFINING ARCHITECTURE
#------------------
class CNN(nn.Module):
    """Regress over image features."""
    def __init__(self, hidden=1, drop=0.1, num_channels=1, num_params=1):
        super().__init__()
        
        #input 1x128x128 ---------> output: 2*hiddenx128x128
        self.conv01 = nn.Conv2d(num_channels, 2*hidden, kernel_size=3, stride=1, padding=1,
                               padding_mode='zeros',bias=True)
        self.conv02 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=3, stride=1, padding=1,
                               padding_mode='zeros', bias=True)
        self.conv03 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=2, stride=2, padding=0,
                               padding_mode='zeros', bias=True)
        
        self.Bat01 = nn.BatchNorm2d(2*hidden)
        self.Bat02 = nn.BatchNorm2d(2*hidden)
        self.Bat03 = nn.BatchNorm2d(2*hidden)
        
        #input 2*hiddenx128x128 ------> output: 4*hiddenx64x64
        self.conv11 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv12 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv13 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
                                padding_mode='zeros', bias=True)
        
        self.Bat11 = nn.BatchNorm2d(4*hidden)
        self.Bat12 = nn.BatchNorm2d(4*hidden)
        self.Bat13 = nn.BatchNorm2d(4*hidden)
        
        #input 4*hidden*64*64 -------> output: 8*hiddenx32x32
        self.conv21 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv22 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv23 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
                                padding_mode='zeros', bias=True)
        
        self.Bat21 = nn.BatchNorm2d(8*hidden)
        self.Bat22 = nn.BatchNorm2d(8*hidden)
        self.Bat23 = nn.BatchNorm2d(8*hidden)
        
        #input 8*hiddenx32x32 ---------> output: 16*hiddenx16x16
        self.conv31 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv32 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv33 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
                                padding_mode='zeros', bias=True)
        
        self.Bat31 = nn.BatchNorm2d(16*hidden)
        self.Bat32 = nn.BatchNorm2d(16*hidden)
        self.Bat33 = nn.BatchNorm2d(16*hidden)
        
        #input 16*hiddenx16x16 ---------> output: 32*hiddenx8x8
        self.conv41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv42 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv43 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=2, padding=0,
                                padding_mode='zeros', bias=True)
        
        self.Bat41 = nn.BatchNorm2d(32*hidden)
        self.Bat42 = nn.BatchNorm2d(32*hidden)
        self.Bat43 = nn.BatchNorm2d(32*hidden)
        
        #input 32*hiddenx8x8 -------> output: 64*hiddenx4x4
        self.conv51 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv52 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv53 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=2, stride=2, padding=0,
                                padding_mode='zeros', bias=True)
        
        self.Bat51 = nn.BatchNorm2d(64*hidden)
        self.Bat52 = nn.BatchNorm2d(64*hidden)
        self.Bat53 = nn.BatchNorm2d(64*hidden)
        
        #input 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
        self.conv61 = nn.Conv2d(64*hidden, 128*hidden, kernel_size=2, stride=1, padding=0,
                                padding_mode='zeros', bias=True)
        self.conv62 = nn.Conv2d(128*hidden, 128*hidden, kernel_size=2, stride=1, padding=0,
                                padding_mode='zeros', bias=True)
        self.conv63 = nn.Conv2d(128*hidden, 128*hidden, kernel_size=2, stride=1, padding=0,
                                padding_mode='zeros', bias=True)
        self.Bat61 = nn.BatchNorm2d(128*hidden)
        self.Bat62 = nn.BatchNorm2d(128*hidden)
        self.Bat63 = nn.BatchNorm2d(128*hidden)
        
        self.P0 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        self.FC1 = nn.Linear(128*hidden, 64*hidden)
        self.FC2 = nn.Linear(64*hidden, num_params*2)
        
        self.dropout = nn.Dropout(p=drop)
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, image):
        # print('0', image.shape)
        
        # LAYER 1
        x = self.LeakyReLU(self.conv01(image))
        x = self.LeakyReLU(self.Bat02(self.conv02(x)))
        # print('1', x.shape)
        # x = self.LeakyReLU(self.Bat03(self.conv03(x)))
        # print('1', x.shape)
        
        # LAYER 2
        x = self.LeakyReLU(self.Bat11(self.conv11(x)))
        x = self.LeakyReLU(self.Bat12(self.conv12(x)))
        x = self.LeakyReLU(self.Bat13(self.conv13(x)))
        # print('2', x.shape)
        
        #LAYER 3
        x = self.LeakyReLU(self.Bat21(self.conv21(x)))
        x = self.LeakyReLU(self.Bat22(self.conv22(x)))
        x = self.LeakyReLU(self.Bat23(self.conv23(x)))
        # print('3', x.shape)
        
        #LAYER 4
        x = self.LeakyReLU(self.Bat31(self.conv31(x)))
        x = self.LeakyReLU(self.Bat32(self.conv32(x)))
        x = self.LeakyReLU(self.Bat33(self.conv33(x)))
        # print('4', x.shape)
        
        #LAYER 5
        x = self.LeakyReLU(self.Bat41(self.conv41(x)))
        x = self.LeakyReLU(self.Bat42(self.conv42(x)))
        x = self.LeakyReLU(self.Bat43(self.conv43(x)))
        # print('5', x.shape)
        
        #LAYER 6
        x = self.LeakyReLU(self.Bat51(self.conv51(x)))
        x = self.LeakyReLU(self.Bat52(self.conv52(x)))
        x = self.LeakyReLU(self.Bat53(self.conv53(x)))
        # print('6', x.shape)
        
        #FINAL
        # print('7', x.shape)
        x = self.LeakyReLU(self.Bat61(self.conv61(x)))
        # print('8', x.shape)
        x = self.LeakyReLU(self.Bat62(self.conv62(x)))
        # print('9', x.shape)
        x = self.LeakyReLU(self.Bat63(self.conv63(x)))
        # print('10', x.shape)
        # sys.exit('end')
        
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        # print('8', x.shape)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)
        
        # print('6',hidden,hidden*2, x.shape)
        return x
        
        #enforce the errors to be positive
        # y = torch.clone(x)
        # y[:,hidden:hidden*2] = torch.square(x[:,hidden:hidden*2])# maybe not accurate
        # return y

        
#------------------------
# Testing Setup and Criterion
#------------------------
Ntrain = split_train
Ntest = split_test
mins = []
maxs = []
for i in range(num_params):
    mins.append(np.min(params[:,i]))
    maxs.append(np.max(params[:,i]))

# print('Pretraining with vanilla MSE loss...')

def criterion1(pred, y):
    return F.mse_loss(pred[:,0:num_params], y, reduction="sum")
# print('\nTraining to maximize Gaussian logpdf...')
def criterion2(pred, y):
    return -(-pred[:,num_params:num_params*2] - \
             ((y-(pred[:,0:num_params]))/(torch.exp(pred[:,num_params:num_params*2])+soft))**2/2).sum()
#     return -(-pred[:,1] - ((y-(pred[:,0]))/(torch.exp(pred[:,1])))**2/2).sum()

trloss_rec = []
teloss_rec = []
optimizer = 0
min_change = 1e-3
# best_model_weights = model.state_dict()
checkpoint_array = []

#-----------------------
# Directories
#-----------------------
data_directory = "/projects/kaad8904/CNN_Light/results/" ### where the final data is outputted
figure_directory = data_directory + "figures/"

root_storage = '/projects/kaad8904/CNN_Light/databases/'
sim = sys.argv[4]
channels = ''
param_str = ''
for i in range(num_channels):
    channels = channels + "_" + sys.argv[5+i]
for i in range(num_params):
    param_str = param_str + "_" + param_titles[i]
study_name = sim+channels+param_str+"_Loss"
storage_name_pt = root_storage+study_name+".pt"
root_storage_db = 'sqlite:///'+root_storage
storage_name_db = root_storage_db+study_name+".db"

#-----------------------------
# Running the Model Function
#------------------------------
def objective(trial):
    hidd = trial.suggest_int("hidden",3,6) # Hyper parameters to change
    dr = trial.suggest_float("dr", 0.0,0.3)
    Lr = trial.suggest_float('lr',1e-5,1e-2,log=True)
    W_d = trial.suggest_float('wd',1e-3,1e-1,log=True)
    
    
    model = CNN(hidden=hidd,drop=dr,num_channels=num_channels,num_params=num_params)
    optimizer = optim.Adam(model.parameters(), lr=Lr, weight_decay=W_d)

    

# TODO
    trloss = 0
    teloss = 0
    pred_array = []
    x_array = []
    y_array = []
    N_epochs = 50
    verbose=False
    temp_valoss_min = np.inf
    # patience = 20
    # wait = 0


    Pretrain_epochs = 0
    broken = False

    
    #------------------
    # Pretrain
    #------------------
    
    for i in range(N_epochs): # testing the model every 20 epochs of the pretraining model
        for x, y, sizes, ids in tqdm(train_loader, disable=not verbose):
            model.train() # dropout on
            optimizer.zero_grad()
            pred = model.forward(x)
            for k in range(num_params):
                y[:,k] = (y[:,k]-mins[k])/(maxs[k]-mins[k])
            # y = (y-minimum)/(maximum-minimum)
            loss = criterion1(pred, y) 
            
            loss.backward()
            optimizer.step()

            trloss += np.nanmean(loss.detach().flatten().numpy())
        with torch.no_grad():
            for x, y, sizes, ids in tqdm(test_loader, disable=not verbose):
                model.eval() # dropout off

                pred = model.forward(x)
                for k in range(num_params):
                    y[:,k] = (y[:,k]-mins[k])/(maxs[k]-mins[k])
                # y = (y-minimum)/(maximum-minimum)
                loss = criterion1(pred, y)

                teloss += np.nanmean(loss.detach().flatten().numpy())

        trloss /= Ntrain
        teloss /= Ntest

        trloss_rec.append(trloss)
        teloss_rec.append(teloss)
        if(i%10==0):
            print(f"epoch: {i+1}/{N_epochs}; train loss: {trloss:.3e}; test loss: {teloss:.3e}")

        # if(teloss_rec[Pretrain_epochs + i] < valoss_min*(1-min_change)):
        # print("temp: ",len(teloss_rec),Pretrain_epochs,i)
        if(teloss_rec[len(teloss_rec)-1] < temp_valoss_min):
            temp_valoss_min = teloss_rec[len(teloss_rec)-1]
            best_model_weights = deepcopy(model.state_dict())
            broken=True


    #----------------------------
    # training
    #---------------------------
    
    N_epochs = 50
    Pretrain_epochs = N_epochs
    if(broken):
        model.load_state_dict(best_model_weights)
    print(f'broken = {broken}')
    broken=False

    for i in range(N_epochs): 
        for x, y, sizes, ids in tqdm(train_loader, disable=not verbose):
            model.train() # dropout on
            optimizer.zero_grad()
            pred = model.forward(x)
            for k in range(num_params):
                y[:,k] = (y[:,k]-mins[k])/(maxs[k]-mins[k])
            # y = (y-minimum)/(maximum-minimum)
            loss = criterion2(pred, y) 

            loss.backward()
            optimizer.step()

            trloss += np.nanmean(loss.detach().flatten().numpy())
        teloss = 0
        with torch.no_grad():
            for x, y, sizes, ids in tqdm(test_loader, disable=not verbose):
                model.eval() # dropout off

                pred = model.forward(x)
                for k in range(num_params):
                    y[:,k] = (y[:,k]-mins[k])/(maxs[k]-mins[k])
                # y = (y-minimum)/(maximum-minimum)
                loss = criterion2(pred, y)

                teloss += np.nanmean(loss.detach().flatten().numpy())

        trloss /= Ntrain
        teloss /= Ntest

        trloss_rec.append(trloss)
        teloss_rec.append(teloss)
        if(i%10==0):
            print(f"epoch: {i+1}/{N_epochs}; train loss: {trloss:.3e}; test loss: {teloss:.3e}")

        # if(teloss_rec[Pretrain_epochs + i] < valoss_min*(1-min_change)):
        # print("temp: ",len(teloss_rec),Pretrain_epochs,i)
        if(teloss_rec[len(teloss_rec)-1] < temp_valoss_min):
            temp_valoss_min = teloss_rec[len(teloss_rec)-1]
            best_model_weights = deepcopy(model.state_dict())
            broken=True
        # else:
        #     wait +=1
    if(broken):
        model.load_state_dict(best_model_weights)
    print(f'broken = {broken}')
    
    #----------------------------
    # Full test for Record and Optuna
    #----------------------------
    
    ys = []
    preds = []
    ids = []
    with torch.no_grad(): 
        for x, y, sizes, ide in tqdm(all_loader,disable=True):
            model.eval()
            pred = model.forward(x)
            ys.append(y)
            for i in range(num_params):
                pred[:,i] = pred[:,i]*(maxs[i]-mins[i]) + mins[i]
                pred[:,i+num_params] += np.log(maxs[i] - mins[i]) # gaussian
                # pred[:,i+num_params] *= (maxs[i] - mins[i])**2 # moment
            preds.append(pred)
            ids.append(ide)
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    ids = np.concatenate(ids)
    
    slope = []
    intercept = []
    r_value = []
    p_value = []
    std_err = []
    scatter = []
    
    for i in range(num_params):
        holder = scipy.stats.linregress(ys[:,i], preds[:,i])
        slope.append(holder[0])
        intercept.append(holder[1])
        r_value.append(holder[2])
        p_value.append(holder[3])
        std_err.append(holder[4])
        
        scatter.append(np.nanstd(preds[:,i]-ys[:,i]))
        
        print(f'Parameter: ',param_titles[i])
        print(f'slope = {slope[i]:.3f},intercept = {intercept[i]:.3f},r_value = {r_value[i]:.3f},p_value = {p_value[i]:.3f},std_err = {std_err[i]:.3f},coefficient of correlation = {r_value[i]**2:.3f}')
        print(f'scatter: {scatter[i]:.3f} dex ')
        
    
    #----------------------------
    # Updating model files
    #----------------------------
    epoch = 0
    starter = 0 ### where the current run technically starts in the database
    same = False ### whether or not the old best trial is the same as the new best trial

    if(len(study.get_trials())>1):
        if(temp_valoss_min<study.best_trial.value):
            epoch = len(study.get_trials())-1
        else:
            checkpoint = torch.load(storage_name_pt)
            hidd = checkpoint['hidden']
            model = CNN(hidden=hidd,drop=dr,num_channels=num_channels,num_params=num_params)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = int(checkpoint['epoch'])
        
        starter = len(study.get_trials())-1 # the number of trials that currently exist
        
    checkpoint_new = {
        'epoch' : epoch,
        'model' : deepcopy(model.state_dict()),
        'optimizer' : deepcopy(optimizer.state_dict()),
        'hidden' : hidd,
        'starter' :  starter
    }
    torch.save(checkpoint_new,storage_name_pt) ### saving whatever comes out
    
    checkpoint_array.append(checkpoint_new)
    
    #--------------------------
    # PLOTTING
    #--------------------------
    for i in range(num_params):
        f, ax = plt.subplots()

        ax.errorbar(ys[:,i],preds[:,i],yerr = np.exp(preds[:,i+num_params]+soft), fmt = '.', alpha = 0.1, color = "Green") # Gaussian PDF
        # ax.errorbar(ys,preds[:,0],yerr = np.sqrt(preds[:,1]), fmt = '.', alpha = 0.1, color = "Green") # Moment network

        window_range = (maxs[i]-mins[i])*0.25/2 # This defines how much space we want around the edges of the graph
        # using minimum and maximum values of the parameter
        ax.plot([mins[i]-window_range,maxs[i]+window_range],[mins[i]-window_range,maxs[i]+window_range],'k--',label="r_value=1")
        x_linreg = np.linspace(mins[i]-window_range,maxs[i]+window_range)
        ax.plot(x_linreg,slope[i]*x_linreg+intercept[i],"r--",label=f"linear regression, r={r_value[i]:.3f}")
        ax.set_xlim(mins[i]-window_range,maxs[i]+window_range*1.1)
        ax.set_ylim(mins[i]-window_range,maxs[i]+window_range*1.1)

        sns.histplot(          # for plotting density map
            x=ys[:,i], y=preds[:,i],
            ax=ax,
            cbar=True,
            alpha=0.5
        )

        if(len(study.get_trials())>1):
            lr_found = study.best_trial.params['lr']
            wd_found = study.best_trial.params['wd']
        else:
            lr_found = Lr
            wd_found = W_d
        ax.title.set_text(f'CNN lr={lr_found:.2e}, wd={wd_found:.2e}')
        ax.set_xlabel(param_titles[i]+'_true')
        ax.set_ylabel(param_titles[i]+'_pred')
        ax.legend()
        plt.show()
        f.tight_layout();
        f.savefig(figure_directory + "Trial_"+trial_number+"_"+param_titles[i]+"_Loss_TvP.png");
        f.savefig("Trial_"+trial_number+"_"+param_titles[i]+"_Loss_TvP.png")
        plt.close(f);
    
    return temp_valoss_min # line fitting




##################RUN MODEL#######################
study = optuna.create_study(study_name = study_name, storage=storage_name_db, direction="minimize"\
                           ,load_if_exists=True) #  and test_loss
num_trials = 8
study.optimize(objective, n_trials=num_trials)

#############LOADING AND SAVING MODELS for testing#############
#---------------------------
# Load model and optim config
#---------------------------
model = CNN(hidden=study.best_trial.params['hidden'],drop=study.best_trial.params['dr'],\
            num_channels=num_channels,num_params=num_params)
optimizer = optim.Adam(model.parameters())


#---------------------------
# checking and instantiating the better model between optuna record and current run
#----------------------------
epoch = 0
starter = 0 ### where the current run technically starts in the database
same = False ### whether or not the old best trial is the same as the new best trial
if(len(glob.glob(storage_name_pt))>0):
    checkpoint = torch.load(storage_name_pt)
    epoch = int(checkpoint['epoch'])
    starter = int(checkpoint['starter'])
    if(int(study.best_trial.number) == epoch):
        same = True

if(same): ### if the best model run hasn't changed
    checkpoint = torch.load(storage_name_pt) # load the file holding the information
    model.load_state_dict(checkpoint['model']) # load the model
    optimizer.load_state_dict(checkpoint['optimizer']) # and the optimizer
    print('same')
else: ### if the best model is not the old one, update, with the index starting at starter
    epoch = int(study.best_trial.number)
    starter = len(study.get_trials())-1
    model.load_state_dict(checkpoint_array[int(study.best_trial.number)-starter]['model'])
    optimizer.load_state_dict(checkpoint_array[int(study.best_trial.number)-starter]['optimizer'])
    print('else')

#----------------------------
# Updating model files
#----------------------------
checkpoint_new = {
    'epoch' : epoch,
    'model' : deepcopy(model.state_dict()),
    'optimizer' : deepcopy(optimizer.state_dict()),
    'hidden' : study.best_trial.params['hidden'],
    'starter' :  starter
}
torch.save(checkpoint_new,storage_name_pt) ### saving whatever comes out


#-------------------------------------------
# Print the best parameters and best accuracy
#-------------------------------------------
print('Best trial:')
print(f"\tnumber: {study.best_trial.number}")
print(f"\tparams: {study.best_trial.params}")
print(f"\tvalues: {study.best_trial.values}")
# trial = study.best_trial # single parameter? # UNNECESSARY BUT GOOD REFERENCE
# print('  teloss: {}'.format(trial.value[0]))
# print('  teloss: {}'.format(trial.values[0]))
# print('  Training Time: {}'.format(trial.values[1]))
# print('  Params: ')
# for key, value in trial.params.items():
#     print('    {}: {}'.format(key, value))

#############TESTING##############

#-----------------------------
#TEST AND TRAIN LOSS PLOTTING
#-----------------------------
plt.figure(figsize=(10,10),facecolor=(1,1,1))
plt.plot(1+np.arange(len(trloss_rec)), trloss_rec, label='train')
plt.plot(1+np.arange(len(teloss_rec)), teloss_rec, label='test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend();
plt.savefig(figure_directory + "Trial_"+trial_number+"_Loss_Loss.png")
plt.savefig("Trial_"+trial_number+"_Loss_Loss.png")


#-------------------------------
#FULL DATA TEST AND PLOTTING
#-------------------------------
ys = []
preds = []
ids = []

with torch.no_grad():
    for x, y, sizes, ide in tqdm(all_loader):
        model.eval()
#         x = x.unsqueeze(1) # 1 Channel only
        pred = model.forward(x)
        ys.append(y)
        for i in range(num_params):
                pred[:,i] = pred[:,i]*(maxs[i]-mins[i]) + mins[i]
                pred[:,i+num_params] += np.log(maxs[i] - mins[i]) # gaussian
                # pred[:,i+num_params] *= (maximum - minimum)**2 # moment
        preds.append(pred)
        ids.append(ide)
ys = np.concatenate(ys)
preds = np.concatenate(preds)
ids = np.concatenate(ids)


#------------------------
# Statistical regressions
#------------------------
slope = []
intercept = []
r_value = []
p_value = []
std_err = []
scatter = []

for i in range(num_params):
    holder = scipy.stats.linregress(ys[:,i], preds[:,i])
    slope.append(holder[0])
    intercept.append(holder[1])
    r_value.append(holder[2])
    p_value.append(holder[3])
    std_err.append(holder[4])

    scatter.append(np.std(preds[:,i]-ys[:,i]))
    
    
#--------------------------
# PLOTTING
#--------------------------
for i in range(num_params):
    f, ax = plt.subplots()

    ax.errorbar(ys[:,i],preds[:,i],yerr = np.exp(preds[:,i+num_params]+soft), fmt = '.', alpha = 0.1, color = "Green") # Gaussian PDF
    # ax.errorbar(ys,preds[:,0],yerr = np.sqrt(preds[:,1]), fmt = '.', alpha = 0.1, color = "Green") # Moment network

    window_range = (maxs[i]-mins[i])*0.25/2 # This defines how much space we want around the edges of the graph
    # using minimum and maximum values of the parameter
    ax.plot([mins[i]-window_range,maxs[i]+window_range],[mins[i]-window_range,maxs[i]+window_range],'k--',label="r_value=1")
    x_linreg = np.linspace(mins[i]-window_range,maxs[i]+window_range)
    ax.plot(x_linreg,slope[i]*x_linreg+intercept[i],"r--",label=f"linear regression, r={r_value[i]:.3f}")
    ax.set_xlim(mins[i]-window_range,maxs[i]+window_range*1.1)
    ax.set_ylim(mins[i]-window_range,maxs[i]+window_range*1.1)
        
    sns.histplot(          # for plotting density map
        x=ys[:,i], y=preds[:,i],
        ax=ax,
        cbar=True,
        alpha=0.5
    )

    lr_found = study.best_trial.params['lr']
    wd_found = study.best_trial.params['wd']
    ax.title.set_text(f'CNN lr={lr_found:.2e}, wd={wd_found:.2e}')
    ax.set_xlabel(param_titles[i]+'_true')
    ax.set_ylabel(param_titles[i]+'_pred')
    ax.legend()
    plt.show()
    f.tight_layout()
    f.savefig(figure_directory + "Trial_"+trial_number+"_"+param_titles[i]+"_Loss_TvP.png");
    f.savefig("Trial_"+trial_number+"_"+param_titles[i]+"_Loss_TvP.png");

#-------------------------------
# OUTPUTTING PREDICTION,ERROR, & TRUTH DATA
#-------------------------------
data_output_dict = {}
for i in range(len(param_titles)):
    data_output_dict[param_titles[i]+'_Pred'] = preds[:,i]
    data_output_dict[param_titles[i]+'_Error'] = preds[:,i+num_params]
    data_output_dict[param_titles[i]+'_Truth'] = ys[:,i]
data_output_dict['ID'] = ids

data_frame = pd.DataFrame(data=data_output_dict)
data_frame.to_csv(data_directory+"CNN_Light_Trial_"+trial_number+"_data_Loss.csv")

# loss_output = open(data_directory+"CNN_Light_Loss_"+ date +"_"+trial_number+".txt",'w')
# loss_output.write("epoch\ttraining loss\ttesting loss\n")
# for i in range(len(trloss_rec)):
#     loss_output.write(f'{i+1}\t{trloss_rec[i]:.3f}\t{teloss_rec[i]:.3f}\n')
# loss_output.close()


#--------------------------
# FINAL DIAGNOSTIC PRINTING
#--------------------------
print("---------------------------------------")
for i in range(num_params):
    print('Parameter: ',param_titles[i])
    print(f'slope = {slope[i]:.3f},intercept = {intercept[i]:.3f},r_value = {r_value[i]:.3f},p_value = {p_value[i]:.3f},std_err = {std_err[i]:.3f},coefficient of correlation = {r_value[i]**2:.3f}')
    print(f'scatter: {scatter[i]:.3f} dex ')
    print("---------------------------------------")
print(model)

print("--- %s seconds ---" % (time.time() - start_time))

# study.best_trial