import numpy as np
import time, sys, os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as axgrid
from accuracy import *
plt.rcParams["figure.dpi"] = 300
plt.rcParams['text.usetex'] = True

# ================================================
# Sys.arvg
# ================================================

if(len(sys.argv)<5):
    print("Usage CGM_plotting_diff_sims_panel.py 1) field_type 2) sim_set 3) sim_z 4) i 5) obs_limit 6) trial number")    
    exit()

field_type = sys.argv[1] # Note, fields divided by _ not ,
#simulation_train = sys.argv[2] # e.g. IllustrisTNG
#simulation_test = sys.argv[3] # e.g. any other simulation
sim_set = sys.argv[2] # e.g. CV
sim_z = sys.argv[3] # e.g. 0.00
i = int(sys.argv[4]) # corresponds to parameter index
if(len(sys.argv)>5):
    obs_limit = sys.argv[5] # Leave blank or '' if 6th argument.  
else:
    obs_limit = ''
if(len(sys.argv)>6):
    trial = int(sys.argv[6])
else:
    trial = -1 # This will break for plotting loss function.  

print("obs_limit= ", obs_limit)

#print("Simulation Train:", simulation_train)
#print("Simulation Test:", simulation_test)

# ================================================
# Getting Files
# ================================================

sims = ['IllustrisTNG', 'SIMBA', 'Astrid']

def get_paths(train, test):
    
    root_dir = '/home/ng474/CMD_py/results/train_%s_test_%s'%(train, test)
    loss_dir = '/home/ng474/CMD_py/full_halos/full_halo_%s/losses_%s'%(field_type, train)
    monopole1 = True
    monopole2 = True
    smoothing1 = 0
    smoothing2 = 0
    arch = 'hp3'
    suffix1 = ''
    if not(monopole1):  suffix1 = '%s_no_monopole'%suffix1
    suffix2 = ''
    if not(monopole2):  suffix2 = '%s_no_monopole'%suffix2

    results_file = 'results_train_%s_%s%s_all_steps_500_500_%s_smoothing_%d%s_test_%s_%s%s_all_steps_500_500_%s_smoothing_%d%s_%s.txt'%(train,field_type,obs_limit,arch,smoothing1,suffix1,test,field_type,obs_limit,arch,smoothing2,suffix2,sim_set)

    loss_file = 'loss_%s%s_%d_all_steps_500_500_%s_smoothing_%d.txt'%(field_type,suffix1,trial,arch,smoothing1)

    run_name = '{}_train_{}_test_{}_{}_z={}{}'.format(field_type, train, test, sim_set, sim_z, obs_limit)
    run_dir = root_dir

    return root_dir, loss_dir, results_file, loss_file, run_dir

# ================================================

# TT: Train on IllustrisTNG, Test on IllustrisTNG

root_dir_TT, loss_dir_TT, results_file_TT, loss_file_TT, run_dir_TT = get_paths(sims[0], sims[0]) # train TNG, test TNG
root_dir_TS, loss_dir_TS, results_file_TS, loss_file_TS, run_dir_TS = get_paths(sims[0], sims[1]) # train TNG, test SIMBA
root_dir_TA, loss_dir_TA, results_file_TA, loss_file_TA, run_dir_TA = get_paths(sims[0], sims[2]) # train TNG, test Astrid

root_dir_ST, loss_dir_ST, results_file_ST, loss_file_ST, run_dir_ST = get_paths(sims[1], sims[0]) # train SIMBA, test TNG
root_dir_SS, loss_dir_SS, results_file_SS, loss_file_SS, run_dir_SS = get_paths(sims[1], sims[1]) # train SIMBA, test SIMBA
root_dir_SA, loss_dir_SA, results_file_SA, loss_file_SA, run_dir_SA = get_paths(sims[1], sims[2]) # train SIMBA, test Astrid

root_dir_AT, loss_dir_AT, results_file_AT, loss_file_AT, run_dir_AT = get_paths(sims[2], sims[0]) # train Astrid, test TNG
root_dir_AS, loss_dir_AS, results_file_AS, loss_file_AS, run_dir_AS = get_paths(sims[2], sims[1]) # train Astrid, test SIMBA
root_dir_AA, loss_dir_AA, results_file_AA, loss_file_AA, run_dir_AA = get_paths(sims[2], sims[2]) # train Astrid, test Astrid

print(results_file_TT)

# ================================================

if field_type == "HI":
    cm = plt.get_cmap('magma')
if field_type == "XraySoft":
    cm = plt.get_cmap('viridis')
else: 
    cm = plt.get_cmap('plasma')

nprop = 6
orientations = 8
splits = 3
def map_params(train, test):

    a, b, results_file, d, run_dir = get_paths(train, test)

    num_maps = sum(1 for line in open(run_dir + '/' + results_file))
    num_maps_plot = int(num_maps/orientations/splits)
 
    return num_maps, num_maps_plot

n_maps_TT, nmaps_plot_TT = map_params(sims[0], sims[0])
n_maps_TS, nmaps_plot_TS = map_params(sims[0], sims[1])
n_maps_TA, nmaps_plot_TA = map_params(sims[0], sims[2])

n_maps_ST, nmaps_plot_ST = map_params(sims[1], sims[0])
n_maps_SS, nmaps_plot_SS = map_params(sims[1], sims[1])
n_maps_SA, nmaps_plot_SA = map_params(sims[1], sims[2])

n_maps_AT, nmaps_plot_AT = map_params(sims[2], sims[0])
n_maps_AS, nmaps_plot_AS = map_params(sims[2], sims[1])
n_maps_AA, nmaps_plot_AA = map_params(sims[2], sims[2])

# ================================================

minimum = np.array([11.5, 8.0, 0.0, 3.9, -3.6, 0.00])
maximum = np.array([14.3, 12.5, 1.0, 7.6, -1.3, 0.23])
field_renorm = [0,0,0,0,0,0] #was 1.87 Renorm, for Solar abundances assuming Asplund+ 2009 log[Z=0.0134], can switch to K to keV if you want.  
property = ['Mhalo','M_CGM', 'fcool', 'logT', 'logZ', 'fcgm']
field_label = [r'$\log(M_{\rm{halo}})$', r'$\log(\rm{M_{CGM}})$', r'$f_{\rm{cool}}$', r'$\log(\rm{T})$', r'$\log(\rm{Z})$', r'$f_{\rm{cgm}}$']
error_label = [r'$\log(M_{\rm{halo}})$', r'$\log(\rm{M_{CGM}})$', r'$f_{\rm{cool}}$', r'$\log(\rm{T})$', r'$\log(\rm{Z})$', r'$f_{\rm{cgm}}/(\Omega_{\rm{b}}/\Omega_{\rm{M}})$']

unit_label = [r'[${\rm M}_{\odot}$]', '[${\rm M}_{\odot}$]', '', r'[K]', r'[${\rm Z}_{\odot}$]', '']
error_renorm = np.array([1.,1.,1.,1.,1.,0.16])

# ================================================

def get_params_1(train, test, idx):

    # i: 0 Mhalo | 1 Mcgm | 2 fcool | 3 logT | 4 logZ | 5 fcgm

    a, b, results_file, d, run_dir = get_paths(train, test)
    num_maps, num_maps_plot = map_params(train, test)

    params_true = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    params_NN   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    errors_NN   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)

    # Read first 1/orientations of maps.  
    params_true[:,0],params_NN[:,0],errors_NN[:,0] = np.loadtxt(run_dir + '/' + results_file,usecols=(0,6,12),unpack=True,max_rows=num_maps_plot*splits)
    params_true[:,0] -= field_renorm[0]

    params_true[:,idx],params_NN[:,idx],errors_NN[:,idx] = np.loadtxt(run_dir + '/' + results_file,usecols=(0+i,6+i,12+i),unpack=True,max_rows=num_maps_plot*splits)
    params_true[:,idx] -= field_renorm[idx]
    params_NN[:,idx] -= field_renorm[idx]

    indexes = np.arange(num_maps_plot)*splits
    mhalo_array = np.asarray(params_true[indexes,0])

    idx = int(idx)

    pTrue_plot = []
    pNN_plot = []
    eNN_plot = []
    for j in range(len(params_true[indexes,idx])):
        index = indexes[j] # need to read indexed split
        pTrue_plot.append(params_true[index,idx])
        pNN_plot.append(params_NN[index, idx])
        eNN_plot.append(errors_NN[index, idx])

    return pTrue_plot, pNN_plot, eNN_plot


def create_subplot(fig, train, test, errs, idx, n_subplot):

    # i: 0 Mhalo | 1 Mcgm | 2 fcool | 3 logT | 4 logZ | 5 fcgm

    a, b, results_file, d, run_dir = get_paths(train, test)
    num_maps, num_maps_plot = map_params(train, test)

    params_true = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    params_NN   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    errors_NN   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)

    params_true_ = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    params_NN_   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    errors_NN_   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)

    # Read first 1/orientations of maps.  
    params_true[:,0],params_NN[:,0],errors_NN[:,0] = np.loadtxt(run_dir + '/' + results_file,usecols=(0,6,12),unpack=True,max_rows=num_maps_plot*splits)
    params_true[:,0] -= field_renorm[0]

    params_true_[:,idx],params_NN_[:,idx],errors_NN_[:,idx] = np.loadtxt(run_dir + '/' + results_file,usecols=(0+i,6+i,12+i),unpack=True,max_rows=num_maps_plot*splits)
    params_true_[:,idx] -= field_renorm[idx]
    params_NN_[:,idx] -= field_renorm[idx]

    indexes = np.arange(num_maps_plot)*splits
    mhalo_array = np.asarray(params_true[indexes,0])

    idx = int(idx)

    #pTrue_plot = []
    #pNN_plot = []
    #eNN_plot = []
    #fig, ax = plt.subplots()
    
    all_RMSE_values = []
    all_R2_values = []
    all_rmerr_values = []
    all_chi2_values = []
    
    ax = fig.add_subplot(3, 3, n_subplot)
    for j in range(len(params_true[indexes,idx])):
        index = indexes[j] # need to read indexed split
        #pTrue_plot.append(params_true[index,idx])
        #pNN_plot.append(params_NN[index, idx])
        #eNN_plot.append(errors_NN[index, idx])
        RMSE = rmse(params_true_[index,idx], params_NN_[index,idx])
        R2 = r2_score(params_true_[:, idx] - field_renorm[idx] , params_NN_[:, idx] - field_renorm[idx] ) #R_squared(params_true[index,i], params_NN_[index,i])
        RM_err = rel_mean_err(params_true_[:, idx] - field_renorm[idx], params_NN_[:, idx] - field_renorm[idx])
        chi2 = chi_squared(params_true_[:, idx] - field_renorm[idx], params_NN_[:, idx] - field_renorm[idx])
        all_RMSE_values.append(RMSE)
        all_R2_values.append(R2)
        all_rmerr_values.append(RM_err)
        all_chi2_values.append(chi2)
        ax.errorbar(params_true[index,idx], params_NN[index,idx], errors_NN[index,idx],
                    linestyle='None', lw=1, fmt='o', ms=2, elinewidth=1, capsize=0, c=cm((params_true[index,0]-minimum[0])/(maximum[0]-minimum[0])), zorder=int(params_true[index,0]*100))

    mean_RMSE = np.mean(all_RMSE_values)
    mean_R2 = np.mean(all_R2_values)
    mean_RMerr = np.mean(all_rmerr_values)
    mean_chi2 = np.mean(all_chi2_values)
    line_RMSE, = ax.plot([],[],label=r'$\rm{RMSE}\ $'+ r'$ = {:.3f}$'.format(mean_RMSE), color='none', markerfacecolor='none')
    line_R2, = ax.plot([],[],label=r'$R^2\ $'+ r'$ = {:.3f}$'.format(mean_R2), color='none', markerfacecolor='none')
    line_RMerr, = ax.plot([],[],label=r'$\epsilon\ $'+ r'$ = {:.3f}$'.format(mean_RMerr), color='none', markerfacecolor='none')
    line_chi2, = ax.plot([],[],label=r'$\chi^2\ $'+ r'$ = {:.3f}$'.format(mean_chi2), color='none', markerfacecolor='none')
    ax.legend(handles=[line_RMSE, line_R2, line_RMerr, line_chi2], handlelength=0, handletextpad=0)

    return ax

def create_subplot_2(fig, train, test, errors, idx, k):

    # i: 0 Mhalo | 1 Mcgm | 2 fcool | 3 logT | 4 logZ | 5 fcgm
    
    a, b, results_file, d, run_dir = get_paths(train, test)
    num_maps, num_maps_plot = map_params(train, test)

    params_true = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    params_NN   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    errors_NN   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)

    # Read first 1/orientations of maps.  
    params_true[:,0],params_NN[:,0],errors_NN[:,0] = np.loadtxt(run_dir + '/' + results_file,usecols=(0,6,12),unpack=True,max_rows=num_maps_plot*splits)
    params_true[:,0] -= field_renorm[0]

    params_true[:,idx],params_NN[:,idx],errors_NN[:,idx] = np.loadtxt(run_dir + '/' + results_file,usecols=(0+idx,6+idx,12+idx),unpack=True,max_rows=num_maps_plot*splits)
    params_true[:,idx] -= field_renorm[idx]
    params_NN[:,idx] -= field_renorm[idx]

    indexes = np.arange(num_maps_plot)*splits
    mhalo_array = np.asarray(params_true[indexes,0])

    idx = int(idx)
    #ax = fig.add_subplot(3, 3, k)
    fig, ax = plt.subplots()
    for j in range(len(params_true[indexes,idx])):
        index = indexes[j] # need to read indexed split
        ax.errorbar(params_true[index,idx], params_NN[index,idx], errors_NN[index,idx], 
                    linestyle='None', lw=1, fmt='o', ms=2, elinewidth=1, capsize=0, c=cm((params_true[index,0]-minimum[0])/(maximum[0]-minimum[0])), zorder=int(params_true[index,0]*100))
                
        if k == 0:
            ax.set_title('Test IllustrisTNG', fontsize=18)
            ax.set_xticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.set_ylabel('Train IllustrisTNG', fontsize=18)
        if k == 1:
            ax.set_title('Test SIMBA')
            ax.set_xticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.set_yticks([])
            ax.yaxis.set_tick_params(labelleft=False)
        if k == 2: 
            ax.set_title('Test Astrid')
            ax.set_xticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.set_yticks([])
            ax.yaxis.set_tick_params(labelleft=False)
        if k == 3: 
            ax.set_ylabel('Train SIMBA', fontsize=18)
            ax.set_xticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
        if k == 4:
            ax.set_xticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.set_yticks([])
            ax.yaxis.set_tick_params(labelleft=False)
        if k == 5: 
            ax.set_xticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.set_yticks([])
            ax.yaxis.set_tick_params(labelleft=False)
        if k == 6: 
           ax.set_ylabel('Train Astrid', fontsize=18)
           if idx == 0:
               ax.set_xlabel(r'$\log(M_{\rm{halo}})$')
           else:
               ax.set_xlabel('param placeholder')
        if k == 7:
           ax.set_yticks([])
           ax.yaxis.set_tick_params(labelleft=False)
        if k == 8:
           ax.set_yticks([])
           ax.yaxis.set_tick_params(labelleft=False)
        
    return ax

# ================================================

x_TT, y_TT, e_TT = get_params_1(sims[0],sims[0],i)
x_TS, y_TS, e_TS = get_params_1(sims[0],sims[1],i)
x_TA, y_TA, e_TA = get_params_1(sims[0],sims[2],i)

x_ST, y_ST, e_ST = get_params_1(sims[1],sims[0],i)
x_SS, y_SS, e_SS = get_params_1(sims[1],sims[1],i)
x_SA, y_SA, e_SA = get_params_1(sims[1],sims[2],i)

x_AT, y_AT, e_AT = get_params_1(sims[2],sims[0],i)
x_AS, y_AS, e_AS = get_params_1(sims[2],sims[1],i)
x_AA, y_AA, e_AA = get_params_1(sims[2],sims[2],i)

x_list = [[x_TT],[x_TS],[x_TA],[x_ST],[x_SS],[x_SA],[x_AT],[x_AS],[x_AA]]
y_list = [[y_TT],[y_TS],[y_TA],[y_ST],[y_SS],[y_SA],[y_AT],[y_AS],[y_AA]]
err_list = [[e_TT],[e_TS],[e_TA],[e_ST],[e_SS],[e_SA],[e_AT],[e_AS],[e_AA]]

# ================================================
#print('colors:',type(colors), cm(colors[0:]))


#fig, axs = plt.subplots(3,3,figsize=(10,10))

#ax_object = get_params_1(sims[0],sims[0],i)
#ax_object.figure.savefig('/home/ng474/CMD_py/panel_testing.png')

fig = plt.figure(figsize=(10, 8))

# Call the create_subplot function for each subplot
ax1 = create_subplot(fig, sims[0], sims[0], err_list[0], i, 1)
ax2 = create_subplot(fig, sims[0], sims[1], err_list[1], i, 2)
ax3 = create_subplot(fig, sims[0], sims[2], err_list[2], i, 3)
ax4 = create_subplot(fig, sims[1], sims[0], err_list[3], i, 4)
ax5 = create_subplot(fig, sims[1], sims[1], err_list[4], i, 5)
ax6 = create_subplot(fig, sims[1], sims[2], err_list[5], i, 6)
ax7 = create_subplot(fig, sims[2], sims[0], err_list[6], i, 7)
ax8 = create_subplot(fig, sims[2], sims[1], err_list[7], i, 8)
ax9 = create_subplot(fig, sims[2], sims[2], err_list[8], i, 9)

# Adjust spacing between subplots
fig.tight_layout()

# Save the subplots to a file
fig.savefig('/home/ng474/CMD_py/panel_subplots_%s.png'%i)

'''
fig = plt.figure()

# Add subplots dynamically
for n in range(1, 10):
    x = x_list[n-1]
    y = y_list[n-1]
    y_err = err_list[n-1]
    fig = create_subplot(fig, x, y, y_err, i, n)

fig.tight_layout()
fig.savefig('/home/ng474/CMD_py/panel_testing.png')
'''


#ax[0,0,0].set_xlabel(r'$\rm{Truth}$',fontsize=22) # was 20
#ax[0,0,0].set_ylabel(r'$\rm{Inference}$',fontsize=22)
#ax[0,0,0].tick_params(axis='both', direction='in', which='major', labelsize=15)
'''
axs[0,0].errorbar(x_TT, y_TT, e_TT, linestyle='None', lw=1, fmt='o', ms=2, elinewidth=1, capsize=0, c=colors)
axs[0,0].set_xticks([])
axs[0,0].xaxis.set_tick_params(labelbottom=False)
axs[0,0].set_ylabel('Train IllustrisTNG', fontsize=18)
axs[0,0].set_title('Test IllustrisTNG', fontsize=18)

axs[0,1].errorbar(x_TS, y_TS, e_TS, linestyle='None', lw=1, fmt='o', ms=2, elinewidth=1, capsize=0)
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[0,1].xaxis.set_tick_params(labelbottom=False)
axs[0,1].yaxis.set_tick_params(labelleft=False)
axs[0,1].set_title('Test SIMBA', fontsize=18)

axs[0,2].errorbar(x_TA, y_TA, e_TA, linestyle='None', lw=1, fmt='o', ms=2, elinewidth=1, capsize=0)
axs[0,2].set_xticks([])
axs[0,2].set_yticks([])
axs[0,2].xaxis.set_tick_params(labelbottom=False)
axs[0,2].yaxis.set_tick_params(labelleft=False)
axs[0,2].set_title('Test Astrid', fontsize=18)


axs[1,0].errorbar(x_ST, y_ST, e_ST, linestyle='None', lw=1, fmt='o', ms=2, elinewidth=1, capsize=0)
axs[1,0].set_xticks([])
axs[1,0].xaxis.set_tick_params(labelbottom=False)
axs[1,0].set_ylabel('Train SIMBA', fontsize=18)

axs[1,1].errorbar(x_SS, y_SS, e_SS, linestyle='None', lw=1, fmt='o', ms=2, elinewidth=1, capsize=0)
axs[1,1].set_xticks([])
axs[1,1].set_yticks([])
axs[1,1].xaxis.set_tick_params(labelbottom=False)
axs[1,1].yaxis.set_tick_params(labelleft=False)

axs[1,2].errorbar(x_SA, y_SA, e_SA, linestyle='None', lw=1, fmt='o', ms=2, elinewidth=1, capsize=0)
axs[1,2].set_xticks([])
axs[1,2].set_yticks([])
axs[1,2].xaxis.set_tick_params(labelbottom=False)
axs[1,2].yaxis.set_tick_params(labelleft=False)


axs[2,0].errorbar(x_AT, y_AT, e_AT, linestyle='None', lw=1, fmt='o', ms=2, elinewidth=1, capsize=0)
axs[2,0].set_ylabel('Train Astrid', fontsize=18)

axs[2,1].errorbar(x_AS, y_AS, e_AS, linestyle='None', lw=1, fmt='o', ms=2, elinewidth=1, capsize=0)
axs[2,1].set_yticks([])
axs[2,1].yaxis.set_tick_params(labelleft=False)

axs[2,2].errorbar(x_AA, y_AA, e_AA, linestyle='None', lw=1, fmt='o', ms=2, elinewidth=1, capsize=0)
axs[2,2].set_yticks([])
axs[2,2].yaxis.set_tick_params(labelleft=False)


fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('panel_plots/plot_{}_{}_{}_z={}{}.trial_{}.png'.format(property[i],field_type, sim_set, sim_z, obs_limit,trial))
'''
'''
print('run_dir', run_dir)
params_true[:,0],params_NN[:,0],errors_NN[:,0] = np.loadtxt(run_dir + '/' + results_file,usecols=(0,6,12),unpack=True,max_rows=num_maps_plot*splits) # Read first 1/orientations of maps.  
params_true[:,0] -= field_renorm[0] 

# np.loadtxt('{}_{}_dataset.txt'.format(field_type, simulation),usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17),unpack=True

print("len(params_true[:,0])= ", len(params_true[:,0]))

indexes = np.arange(num_maps_plot)*splits

for i in range(nprop):
    print("i= ", i)
    params_true[:,i],params_NN[:,i],errors_NN[:,i] = np.loadtxt(run_dir + '/' + results_file,usecols=(0+i,6+i,12+i),unpack=True,max_rows=num_maps_plot*splits)

    params_true[:,i] -= field_renorm[i]
    params_NN[:,i] -= field_renorm[i]

    mhalo_array = np.asarray(params_true[indexes,0])

    fig=plt.figure(figsize=(7.5,7.5))
    ax = fig.add_subplot(111) 

    ax.set_xlabel(r'$\rm{Truth}$',fontsize=22) # was 20
    ax.set_ylabel(r'$\rm{Inference}$',fontsize=22)
    ax.tick_params(axis='both', direction='in', which='major', labelsize=15)

    for j in range(len(params_true[indexes,i])):
        index = indexes[j] # need to read indexed split
        errorbar = ax.errorbar(params_true[index,i], params_NN[index,i], errors_NN[index,i], 
                    linestyle='None', lw=1, fmt='o', ms=2, elinewidth=1, capsize=0, c=cm((params_true[index,0]-minimum[0])/(maximum[0]-minimum[0])), zorder=int(params_true[index,0]*100))
        #print(i,j,index,params_true[index,i], params_NN[index,i])

    # Make colorbar.  if statements chooses what plots.  If set to i==0, then only halo plot gets it, but it may be more sensible to put elsewhere.
    if(i==0): 
        div = axgrid.make_axes_locatable(ax)
        dummydata = ax.scatter([minimum[i],maximum[i]], [minimum[i],maximum[i]], c=[minimum[0],maximum[0]], cmap=cm,s=0.001)
        cax = div.append_axes("right", size="5%", pad=-0.2, zorder=20)
        cb = fig.colorbar(dummydata,cax=cax)
        cb.ax.tick_params(labelsize=15) # was 12
        cb.ax.set_xlabel(r'$\log(M_{halo}/M_{\odot})$',labelpad=16,fontsize=18) # was 14
    substring = "_"
    str_field_type = str(field_type)
    multifield = str_field_type.split('_')
    #print('multifield:', multifield)
    multifield_replace = [s.replace('XraySoft', 'Xray') for s in multifield]
    field_type_replace = field_type.replace("XraySoft", "Xray")

    #print('multifield_replace:',multifield_replace)
    #print('field_type_replace:',field_type_replace)
    if(len(multifield)>1):    
        #str_field_type = str(field_type)
        #multifield = str_field_type.split('_')
        #print('multifield:', multifield)
        ax.set_title(field_label[i]+ " " + r'$\rm{%s}\ \rm{%s}\ \rm{Train}\ \rm{%s}\ \rm{Test}\ \rm{%s}$'%(multifield_replace[0], multifield_replace[1], simulation_train, simulation_test), fontsize=22 )
    else: 
        ax.set_title(field_label[i] + " " + r'$\rm{%s}\ \rm{Train}\ \rm{%s}\ \rm{Test}\ \rm{%s}$'%(field_type_replace, simulation_train, simulation_test),fontsize=22)

    ax.plot([minimum[i],maximum[i]], [minimum[i],maximum[i]], color='k')

    if(sys.argv[5]=='.obslimit'):
        ax.text((maximum[i]-minimum[i])*0.02+minimum[i], (maximum[i]-minimum[i])*0.98+minimum[i], r'$\rm{Observational\ Limit}$', fontsize=22)

    #ax.text((maximum[i]-minimum[i])*0.03+minimum[i], (maximum[i]-minimum[i])*0.97+minimum[i],'Trial %d'%trial,fontsize=13)

    ax.set_xlim(minimum[i]-(maximum[i]-minimum[i])*0.1,maximum[i]+(maximum[i]-minimum[i])*0.1) # Unmoving boundaries at +/-10% of range.  
    ax.set_ylim(minimum[i]-(maximum[i]-minimum[i])*0.1,maximum[i]+(maximum[i]-minimum[i])*0.1) # Unmoving boundaries at +/-10% of range. 
    fig.savefig('panel_plots/plot_{}_train_{}_test_{}_{}_{}_z={}{}.trial_{}.png'.format(property[i],field_type, simulation_train, simulation_test, sim_set, sim_z, obs_limit,trial) )
'''
