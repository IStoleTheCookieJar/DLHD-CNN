import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Naomi_Code.accuracy import *
from Naomi_Code.data_cuts import *

####################################################################
"""
!!! Note that I am using Naomi's Code in the Naomi_Code.---- , you will need some form of access to this via the import statements !!!

If all the paths are changed to be correct, and 'CNN_Light_Alpine_inputs.txt' (should be in local directory too)
is in the format of:

254 7_30_2024 4 300 0.2-0.4_ccd 0.4-0.6_ccd 0.7-1.1_ccd 1.4-2.0_ccd 2 M500c f_gas_R500c
255 7_30_2024 4 100 0.2-0.4_ccd 0.4-0.6_ccd 0.7-1.1_ccd 1.4-2.0_ccd 3 M500c f_gas_R500c f_cool_R500c
256 7_30_2024 7 300 0.2-0.4_lem 0.4-0.6_lem 0.73-1.1_lem 1.43-2.0_lem Fe_XVII O_VII O_VIII 2 M500c f_gas_R500c
257 7_30_2024 7 100 0.2-0.4_lem 0.4-0.6_lem 0.73-1.1_lem 1.43-2.0_lem Fe_XVII O_VII O_VIII 3 M500c f_gas_R500c f_cool_R500c

Where the arguments go: 
Trial_number , Date , #_of_Channels , Simulation , Channel_First , ... , Channel Last , #_of_parameters, Parameter_first, ... , Parameter_last

This file will automatically construct the names of the outputted .csv files from the Scatter and Loss Network.

!!! Keep in mind, that data_dir needs to be changed to reflect where the you want to grab the .csv from !!!

This takes three positional arguements:
1. Trial_number you would like to access from 'CNN_Light_Alpine_inputs.txt'
2. Whether you would like 'Scatter' or 'Loss'
3. Colorbar or not? "Y" or "N"
"""

####################################################################
# Necessary if the chosen Trial does not include Mass as a parameter

Names = Names = ["ID","M500c","f_gas_R500c","a_form","conc_200c","shape","Gamma_R500C"\
             ,"Gamma_vir","Log_T","Log_Z","OFe","NeFe","CO","axis placeholder"\
             ,"M500c_","M200c","f_gas_0.5R200c","f_gas_0.7R200c","f_gas_R200c","f_gas_2R200c"]
Dataset_overlord = pd.read_csv("/pl/active/CASA/beop5934/halos/TNG300/Params_Groups_TNG300_z=0.00.ascii_ID_fgas", delimiter = "\s+",names = Names)
DO_new = pd.DataFrame(np.repeat(Dataset_overlord.values,3,axis=0))
DO_new.columns = Dataset_overlord.columns
Names_100 = ["ID","M200c","f_gas_R500c", "a_form","conc_200c","shape","Log_M_star","M500c",
                 "f_gas_0.5R200c","f_gas_0.7R200c","f_gas_R200c","f_gas_2R200c",
                 "Log_Z_50","Log_Z_100","Log_Z_200","OFe_50","OFe_100","OFe_200","f_cool_R500c"]
Dataset_overlord_100 = pd.read_csv("/pl/active/CASA/beop5934/halos/TNG100/Params_TNG100_z=0.00.multiradii.ascii_ID_fgas", delimiter = "\s+",names = Names_100)
DO_100_new = pd.DataFrame(np.repeat(Dataset_overlord_100.values,3,axis=0))
DO_100_new.columns = Dataset_overlord_100.columns
######################################################################

# Reading the input arguement file so that only a trial number needs to be passed
f = open('CNN_Light_Alpine_inputs.txt')
input_array = f.readlines()
output_array = []
n_empt_lin = 0
for i in range(len(input_array)):
    line = input_array[i].split()
    if(len(line) > 0):
        output_array.append(line)
        
# for i in range(len(output_array)): ### You can check what trial numbers and their corresponding combos are here by uncommenting this
#     print(output_array[i])
    
#######################################################################
trial_number = sys.argv[1] # To be replaced with an input argument
tipe = sys.argv[2] # to be replaced with an input argument
index = -1
channels = []
params = []
for i in range(len(output_array)):
    if(int(output_array[i][0]) == int(trial_number)):
        index = i

if(index == -1):
    sys.exit('Trial Number not found, try again')
else:
    date = output_array[index][1]
    num_channels = int(output_array[index][2])
    sim = output_array[index][3]
    for i in range(num_channels):
        channels.append(output_array[index][4+i])
    num_params = int(output_array[index][4+num_channels])
    for i in range(num_params):
        params.append(output_array[index][4+num_channels+1+i])
        
data_dir = "/projects/kaad8904/CNN_Light/results/"
results_dir = data_dir+"figures/"
df = pd.read_csv(data_dir+"CNN_Light_Trial_"+trial_number+"_data_"+tipe+".csv")

###################################################################################
cm = plt.get_cmap("plasma")
for i in range(num_params):
    fig = plt.figure()  #fig = plt.figure(figszie=(xinches,yinches))
    ax = fig.add_subplot()
    
    x = df[df.keys()[3+i*3]].values
    y = df[df.keys()[1+i*3]].values
    y_err = df[df.keys()[2+i*3]].values
    maxs = np.max(x)
    mins = np.min(x)
    window_range = (maxs-mins)*0.25/2
    
    RMSE = rmse(x, y)
    R2 = r2_score(x,y)
    RM_err = rel_mean_err(x,y)
    chi2 = chi_squared(x,y,np.exp(y_err))
    
    RMSE = np.nanmean(RMSE)
    R2 = np.nanmean(R2)
    RM_err = np.nanmean(RM_err)
    chi2 = np.nanmean(chi2)
    
    if(sim == "300"):
        for j in range(len(x)):
            if(j%trim_mhalo_data(DO_new["M500c"][j]) != 0): continue
            ax.errorbar(x[j],y[j],np.exp(y_err[j]),linestyle='None', lw=1, fmt='o', ms=2, \
                        elinewidth=1, capsize=0, alpha=0.1, \
                        c=cm((DO_new["M500c"][j]-np.min(DO_new["M500c"]))/(np.max(DO_new["M500c"])-np.min(DO_new["M500c"]))))

        dummydata = ax.scatter([np.min(DO_new["M500c"]),np.max(DO_new["M500c"])],\
                               [np.min(DO_new["M500c"]),np.max(DO_new["M500c"])],\
                               c=[np.min(DO_new["M500c"]),np.max(DO_new["M500c"])], cmap='plasma',s=0.001)

    elif(sim == "100"):
        for j in range(len(x)):
            if(j%trim_mhalo_data(DO_100_new["M500c"][j]) != 0): continue
            ax.errorbar(x[j],y[j],np.exp(y_err[j]),linestyle='None', lw=1, fmt='o', ms=2, \
                        elinewidth=1, capsize=0, alpha=0.1, \
                        c=cm((DO_100_new["M500c"][j]-np.min(DO_100_new["M500c"]))/(np.max(DO_100_new["M500c"])-np.min(DO_100_new["M500c"]))))

        dummydata = ax.scatter([np.min(DO_100_new["M500c"]),np.max(DO_100_new["M500c"])],\
                               [np.min(DO_100_new["M500c"]),np.max(DO_100_new["M500c"])],\
                               c=[np.min(DO_100_new["M500c"]),np.max(DO_100_new["M500c"])], cmap='plasma',s=0.001)

    ax.plot([mins-window_range,maxs+window_range],[mins-window_range,maxs+window_range],'k--')
    if((sys.argv[3] == "Y") | (sys.argv[3] == 'y')):
        fig.colorbar(dummydata,label="Log(M500c)")
    
    ##########Title construction
    title = f"|TNG{sim}|"
    ccd = 0
    lem_wide = 0
    lem_narrow = 0
    H1 = 0
    for j in range(len(channels)):
        if(channels[j][-3:]=="ccd"):
            ccd+=1
        elif(channels[j][-3:]=="lem"):
            lem_wide+=1
        elif(channels[j][:2]=="H1"):
            H1+=1
        else:
            lem_narrow+=1
    
    if(ccd != 0):
        title = title+f"{ccd} CCD Channels|"
    if(lem_wide != 0):
        title = title+f"{lem_wide} LEM Wide_Band|"
    if(lem_narrow != 0):
        title = title+f"{lem_narrow} LEM Narrow_Band|"
    if(H1==1):
        title = title+"H1|"
    if(H1==2):
        title = title+"H1 and H1 velocity|"
    if(H1==3):
        title = title+"H1,vel,disp|"
    title = title+tipe+"|"
    ###############################
            
    ax.title.set_text(title) # What the input is // Maybe make a key for letters so its not too long, or use all inputs
    ax.set_xlim(mins-window_range,maxs+window_range*1.1)
    ax.set_ylim(mins-window_range,maxs+window_range*1.1)
    ax.set_xlabel(params[i]+" Truth") # Truth of Property
    ax.set_ylabel(params[i]+" Prediction") # Pred of Property
    line_RMSE, = ax.plot([],[],label=r'$\rm{RMSE}\ $'+ r'$ = {:.3f}$'.format(RMSE), color='none', markerfacecolor='none')
    line_R2, = ax.plot([],[],label=r'$R^2\ $'+ r'$ = {:.3f}$'.format(R2), color='none', markerfacecolor='none')
    line_RMerr, = ax.plot([],[],label=r'$\epsilon\ $'+ r'$ = {:.3f}$'.format(RM_err), color='none', markerfacecolor='none')
    line_chi2, = ax.plot([],[],label=r'$\chi^2\ $'+ r'$ = {:.3f}$'.format(chi2), color='none', markerfacecolor='none')
    ax.legend(handlelength=0, handletextpad=0, loc='lower right')
#     plt.show()
    plt.subplots_adjust(left=0.1, bottom=0.1,top=0.9,right=0.85)
    plt.savefig(results_dir+"Trial_"+trial_number+"_"+params[i]+"_"+tipe+"_TvP.png")
    
print("Done!")