import numpy as np
import time, sys, os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as axgrid
from data_cuts import *
from accuracy import *
plt.rcParams["figure.dpi"] = 300
plt.rcParams['text.usetex'] = True

# ================================================
# Sys.arvg
# ================================================

if(len(sys.argv)<4):
    print("Usage CGM_plotting_fields_panel.py 1) sim_set 2) sim_z 3) i 4) obs_limit 5) trial number")    
    exit()

#field_type = sys.argv[1] # Note, fields divided by _ not ,
#simulation_train = sys.argv[2] # e.g. IllustrisTNG
#simulation_test = sys.argv[3] # e.g. any other simulation
sim_set = sys.argv[1] # e.g. CV
sim_z = sys.argv[2] # e.g. 0.00
i = int(sys.argv[3]) # corresponds to parameter index
if(len(sys.argv)>4):
    obs_limit = sys.argv[4] # Leave blank or '' if 6th argument.  
else:
    obs_limit = ''
if(len(sys.argv)>5):
    trial = int(sys.argv[5])
else:
    trial = -1 # This will break for plotting loss function.  

print("obs_limit= ", obs_limit)

#print("Simulation Train:", simulation_train)
#print("Simulation Test:", simulation_test)

# ================================================
# Getting Files
# ================================================

sims = ['IllustrisTNG', 'SIMBA', 'Astrid']

def get_paths(field_type, train, test):
    
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

#if field_type == "HI":
#    cm = plt.get_cmap('magma')
#if field_type == "XraySoft":
#    cm = plt.get_cmap('viridis')
#if field_type == "HI_XraySoft": 
#    cm = plt.get_cmap('plasma')

nprop = 6
orientations = 8
splits = 3
def map_params(field_type, train, test):

    a, b, results_file, d, run_dir = get_paths(field_type, train, test)

    num_maps = sum(1 for line in open(run_dir + '/' + results_file))
    num_maps_plot = int(num_maps/orientations/splits)
 
    return num_maps, num_maps_plot

# ================================================

minimum = np.array([11.5, 8.0, 0.0, 3.9, -3.6, 0.00])
maximum = np.array([14.3, 12.5, 1.0, 7.6, -1.3, 0.23])
field_renorm = [0,0,0,0,0,0] # Z was 1.87 Renorm, for Solar abundances assuming Asplund+ 2009 log[Z=0.0134], can switch to K to keV if you want.  
property = ['Mhalo','M_CGM', 'fcool', 'logT', 'logZ', 'fcgm']
field_label = [r'$\log(M_{\rm{halo}})$', r'$\log(\rm{M_{CGM}})$', r'$f_{\rm{cool}}$', r'$\log(\rm{T})$', r'$\log(\rm{Z})$', r'$f_{\rm{cgm}}$']
error_label = [r'$\log(M_{\rm{halo}})$', r'$\log(\rm{M_{CGM}})$', r'$f_{\rm{cool}}$', r'$\log(\rm{T})$', r'$\log(\rm{Z})$', r'$f_{\rm{cgm}}/(\Omega_{\rm{b}}/\Omega_{\rm{M}})$']

unit_label = [r'[${\rm M}_{\odot}$]', '[${\rm M}_{\odot}$]', '', r'[K]', r'[${\rm Z}_{\odot}$]', '']
error_renorm = np.array([1.,1.,1.,1.,1.,0.16])

if i == 0:
    label = 'Mhalo'
    plt_txt = 'M_{\rm{halo}}'
if i == 1 :
    label = 'Mcgm'
    plt_txt = 'M_{{\rmCGM}}'
if i == 2:
    label = 'fcool'
    plt_txt = 'f_{\rm{cool}}'
if i == 3:
    label = 'logT'
    plt_txt = '\log(T)'
if i == 4:
    label = 'logZ'
    plt_txt = '\log(Z)'
if i == 5:
    label = 'fcgm'
    plt_txt = 'f_{\rm{cgm}}'

# ================================================

def get_params_1(field_type, train, test, i):

    # i: 0 Mhalo | 1 Mcgm | 2 fcool | 3 logT | 4 logZ | 5 fcgm

    #print('field_type:', field_type)
    #print('train:', train, 'test:', test)

    i = int(i)
    a, b, results_file, d, run_dir = get_paths(field_type, train, test)
    num_maps, num_maps_plot = map_params(field_type, train, test)

    params_true = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    params_NN   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    errors_NN   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)

    # Read first 1/orientations of maps.  
    params_true[:,0],params_NN[:,0],errors_NN[:,0] = np.loadtxt(run_dir + '/' + results_file,usecols=(0,6,12),unpack=True,max_rows=num_maps_plot*splits)
    params_true[:,0] -= field_renorm[0]

    params_true[:,i],params_NN[:,i],errors_NN[:,i] = np.loadtxt(run_dir + '/' + results_file,usecols=(0+i,6+i,12+i),unpack=True,max_rows=num_maps_plot*splits)
    params_true[:,i] -= field_renorm[i]
    params_NN[:,i] -= field_renorm[i]

    indexes = np.arange(num_maps_plot)*splits
    mhalo_array = np.asarray(params_true[indexes,0])

    pTrue_plot = []
    pNN_plot = []
    eNN_plot = []

    for j in range(len(params_true[indexes,i])):
        index = indexes[j] # need to read indexed split
        pTrue_plot.append(params_true[index,i])
        pNN_plot.append(params_NN[index, i])
        eNN_plot.append(errors_NN[index, i])

    return pTrue_plot, pNN_plot, eNN_plot


def create_subplot(fig, field_type, train, test, i, n_subplot):

    # i: 0 Mhalo | 1 Mcgm | 2 fcool | 3 logT | 4 logZ | 5 fcgm

    print('field_type:', field_type)
    print('train:', train, 'test:', test)
    
    a, b, results_file, d, run_dir = get_paths(field_type, train, test)
    num_maps, num_maps_plot = map_params(field_type, train, test)
    
    params_true = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    params_NN   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    errors_NN   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)

    params_true_ = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    params_NN_   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
    errors_NN_   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)

    i = int(i)
    # Read first 1/orientations of maps.  
    params_true[:,0],params_NN[:,0],errors_NN[:,0] = np.loadtxt(run_dir + '/' + results_file,usecols=(0,6,12),unpack=True,max_rows=num_maps_plot*splits)
    params_true[:,0] -= field_renorm[0]

    params_true_[:,i],params_NN_[:,i],errors_NN_[:,i] = np.loadtxt(run_dir + '/' + results_file,usecols=(0+i,6+i,12+i),unpack=True,max_rows=num_maps_plot*splits)
    params_true_[:,i] -= field_renorm[i]
    params_NN_[:,i] -= field_renorm[i]
    
    indexes = np.arange(num_maps_plot)*splits
    mhalo_array = np.asarray(params_true[indexes,0])

    if field_type == "HI":
        cm = plt.get_cmap('magma')
    if field_type == "XraySoft":
        cm = plt.get_cmap('viridis')
    if field_type == "HI_XraySoft": 
        cm = plt.get_cmap('plasma')
    
    all_RMSE_values = []
    all_R2_values = []
    all_rmerr_values = []
    all_chi2_values = []
    ax = fig.add_subplot(2, 3, n_subplot)
    for j in range(len(params_true_[indexes,i])):
        index = indexes[j] # need to read indexed split
        if(j%trim_mhalo_data(params_true[index,0]) != 0): continue
        ax.errorbar(params_true_[index,i], params_NN_[index,i], errors_NN_[index,i],
                    linestyle='None', lw=1, fmt='o', ms=2, elinewidth=1, capsize=0, c=cm((params_true[index,0]-minimum[0])/(maximum[0]-minimum[0])), zorder=int(params_true[index,0]*100))
        RMSE = rmse(params_true_[index,i], params_NN_[index,i])
        R2 = r2_score(params_true_[:, i] - field_renorm[i] , params_NN_[:, i] - field_renorm[i] ) #R_squared(params_true[index,i], params_NN_[index,i])
        RM_err = rel_mean_err(params_true_[:, i] - field_renorm[i], params_NN_[:, i] - field_renorm[i])
        chi2 = chi_squared(params_true_[:, i] - field_renorm[i], params_NN_[:, i] - field_renorm[i], errors_NN_[:, i])
        all_RMSE_values.append(RMSE)
        all_R2_values.append(R2)
        all_rmerr_values.append(RM_err)
        all_chi2_values.append(chi2)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
        ax.plot([minimum[i],maximum[i]], [minimum[i],maximum[i]], color='k')
        #ax.set_xlim(minimum[i]-(maximum[i]-minimum[i])*0.1,maximum[i]+(maximum[i]-minimum[i])*0.1) # Unmoving boundaries at +/-10% of range.  
        #ax.set_ylim(minimum[i]-(maximum[i]-minimum[i])*0.1,maximum[i]+(maximum[i]-minimum[i])*0.1) # Unmoving boundaries at +/-10% of range.

        if n_subplot == 1:
            ax.set_title(r'$\rm{IllustrisTNG}$', fontsize=18)
            #ax.set_xticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            if i == 0:
                ax.set_ylabel(r'$\log(M_{\rm{halo, infer}})$', fontsize=18)
                ax.axes.text(11.55, 14, r'$\rm X–ray$', fontsize=16)
            if i == 4:
                ax.set_ylabel(r'$\log(Z_{\rm{cgm, infer}})$', fontsize=18)
                ax.axes.text(-2.8, -1.7, r'$\rm X–ray$', fontsize=16)
                ax.set_ylim(-3, -1.5)
                ax.set_xlim(-3, -1.5)
                ax.set_yticks([-3, -2.5, -2, -1.5])
            if i == 5:
                ax.set_ylabel(r'$f_{\rm{cgm, infer}}$', fontsize=18)
                ax.axes.text(0.01, 0.15, r'$\rm X–ray$', fontsize=16)
                ax.set_xlim(0, 0.17)
                ax.set_ylim(0, 0.17)
                ax.set_yticks([0, 0.05, 0.1, 0.15])
        elif n_subplot == 2:
            ax.set_title(r'$\rm{SIMBA}$', fontsize=18)
            #ax.set_xticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            #ax.set_yticks([])
            ax.yaxis.set_tick_params(labelleft=False)
            if i == 4:
                ax.set_ylim(-3, -1.5)
                ax.set_xlim(-3, -1.5)
            if i == 5:
                ax.set_xlim(0, 0.17)
                ax.set_ylim(0, 0.17)
        elif n_subplot == 3:
            ax.set_title(r'$\rm{Astrid}$', fontsize=18)
            #ax.set_xticks([])
            ax.xaxis.set_tick_params(labelbottom=False)
            #ax.set_yticks([])
            ax.yaxis.set_tick_params(labelleft=False)
            div = axgrid.make_axes_locatable(ax)
            cm = plt.get_cmap('viridis')
            dummydata = ax.scatter([minimum[i],maximum[i]], [minimum[i],maximum[i]], c=[minimum[0],maximum[0]], cmap=cm,s=0.001)
            cax = div.append_axes("right", size="5%", pad=0, zorder=20)
            cb = fig.colorbar(dummydata,cax=cax)
            cb.ax.tick_params(labelsize=15) # was 12
            cb.ax.set_ylabel(r'$\log(M_{\rm{halo}}/M_{\odot})$',labelpad=16,fontsize=18) # was 14
            if i == 4:
                ax.set_ylim(-3, -1.5)
                ax.set_xlim(-3, -1.5)
            if i == 5:
                ax.set_xlim(0, 0.17)
                ax.set_ylim(0, 0.17)
        elif n_subplot == 4:
            if i == 0:
                ax.set_ylabel(r'$\log(M_{\rm{halo, infer}})$', fontsize=18)
                ax.set_xlabel(r'$\log(M_{\rm{halo, truth}})$', fontsize=18)
                ax.axes.text(11.55, 14, r'$\rm{HI}$', fontsize=16)
            if i == 4:
                ax.set_ylabel(r'$\log(Z_{\rm{cgm, infer}})$', fontsize=18)
                ax.set_xlabel(r'$\log(Z_{\rm{cgm, truth}})$', fontsize=18)
                ax.axes.text(-2.8, -1.7, r'$\rm{HI}$', fontsize=16)
                ax.set_ylim(-3, -1.5)
                ax.set_xlim(-3, -1.5)
                ax.set_yticks([-3, -2.5, -2])
                ax.set_xticks([-3, -2.5, -2])
            if i == 5:
                ax.set_ylabel(r'$f_{\rm{cgm, infer}}$', fontsize=18)
                ax.set_xlabel(r'$f_{\rm{cgm, truth}}$', fontsize=18)
                ax.axes.text(0.01, 0.15, r'$\rm{HI}$', fontsize=16)
                ax.set_xlim(0, 0.17)
                ax.set_ylim(0, 0.17)
                ax.set_yticks([0, 0.05, 0.1, 0.15])
                ax.set_xticks([0, 0.05, 0.1, 0.15])
        elif n_subplot == 5:
            #ax.set_yticks([])
            ax.yaxis.set_tick_params(labelleft=False)
            if i == 0:
                ax.set_xlabel(r'$\log(M_{\rm{halo, truth}})$', fontsize=18)
            if i == 4:
                ax.set_xlabel(r'$\log(Z_{\rm{cgm, truth}})$', fontsize=18)
                ax.set_ylim(-3, -1.5)
                ax.set_xlim(-3, -1.5)
                ax.set_yticks([-3, -2.5, -2])
                ax.set_xticks([-3, -2.5, -2])
            if i == 5:
                ax.set_xlabel(r'$f_{\rm{cgm, truth}}$', fontsize=18)
                ax.set_xlim(0, 0.17)
                ax.set_ylim(0, 0.17)
                ax.set_xticks([0, 0.05, 0.1, 0.15])
        elif n_subplot == 6:
            #ax.set_yticks([])
            ax.yaxis.set_tick_params(labelleft=False)
            if i == 0:
                ax.set_xlabel(r'$\log(M_{\rm{halo, truth}})$', fontsize=18)
            if i == 4:
                ax.set_xlabel(r'$\log(Z_{\rm{cgm, truth}})$', fontsize=18)
                ax.set_ylim(-3, -1.5)
                ax.set_xlim(-3, -1.5)
                ax.set_yticks([-3, -2.5, -2])
                ax.set_xticks([-3, -2.5, -2, -1.5])
            if i == 5:
                ax.set_xlabel(r'$f_{\rm{cgm, truth}}$', fontsize=18)
                ax.set_xlim(0, 0.17)
                ax.set_ylim(0, 0.17)
                ax.set_xticks([0, 0.05, 0.1, 0.15])
            div = axgrid.make_axes_locatable(ax)
            cm = plt.get_cmap('magma')
            dummydata = ax.scatter([minimum[i],maximum[i]], [minimum[i],maximum[i]], c=[minimum[0],maximum[0]], cmap=cm,s=0.001)
            cax = div.append_axes("right", size="5%", pad=0, zorder=20)
            cb = fig.colorbar(dummydata,cax=cax)
            cb.ax.tick_params(labelsize=15) # was 12
            cb.ax.set_ylabel(r'$\log(M_{\rm{halo}}/M_{\odot})$',labelpad=16,fontsize=18) # was 14
    mean_RMSE = np.mean(all_RMSE_values)
    mean_R2 = np.mean(all_R2_values)
    mean_RMerr = np.mean(all_rmerr_values)
    mean_chi2 = np.mean(all_chi2_values)
    line_RMSE, = ax.plot([],[],label=r'$\rm{RMSE}\ $'+ r'$ = {:.3f}$'.format(mean_RMSE), color='none', markerfacecolor='none')
    line_R2, = ax.plot([],[],label=r'$R^2\ $'+ r'$ = {:.3f}$'.format(mean_R2), color='none', markerfacecolor='none')
    line_RMerr, = ax.plot([],[],label=r'$\epsilon\ $'+ r'$ = {:.3f}$'.format(mean_RMerr), color='none', markerfacecolor='none')
    line_chi2, = ax.plot([],[],label=r'$\chi^2\ $'+ r'$ = {:.3f}$'.format(mean_chi2), color='none', markerfacecolor='none')
    ax.legend(handles=[line_RMSE, line_R2, line_RMerr, line_chi2], handlelength=0, handletextpad=0, loc='lower right')
    return ax


# ================================================

fig = plt.figure(figsize=(11,8))
# Xray Plots
ax1 = create_subplot(fig, 'XraySoft', sims[0], sims[0], i, 1)
ax2 = create_subplot(fig, 'XraySoft', sims[1], sims[1], i, 2)
ax3 = create_subplot(fig, 'XraySoft', sims[2], sims[2], i, 3)

# HI Plots
ax4 = create_subplot(fig, 'HI', sims[0], sims[0], i, 4)
ax5 = create_subplot(fig, 'HI', sims[1], sims[1], i, 5)
ax6 = create_subplot(fig, 'HI', sims[2], sims[2], i, 6)

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.02, hspace=0.02)
#fig.tight_layout()

# Save the subplots to a file

print('Label:', label)
fig.savefig('/home/ng474/CMD_py/panel_cut_2x3_%s%s.png'%(label,obs_limit))



