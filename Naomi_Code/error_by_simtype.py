import numpy as np
import time, sys, os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as axgrid

plt.rcParams["figure.dpi"] = 300
plt.rcParams['text.usetex'] = True

'''
For now, this script will use XRay for both as a placeholder,
since HI is not finished re-running the training set.

This will need to be modified to take in two separate
input fields, but not as a multifield.

As of 3/25/23, this works for using XRay as both inputs
for the Error by Halo Type plot. The order of the bars
is the updated order: Mhalo, fcgm, T, Mcgm, fcool, Z.

This is the in-progress updated version of the original 
and working error_by_halotype_plot.py script, to now use
two different fields, but not as a multifield, for the 
error by halo type plot.

Adding in a field_type B works still when both fields are XRay.
Need to figure out how to add in an additional trial_number.

As of 5/10/23, modified min/max values for all simulations.

Modified on 5/13/23. This script will produce a plot to focus
on the observational limit runs only, and will use all three
simulations. This only focuses on HI+Xray, since this gives
IllustrisTNG and SIMBA the best chance of the lowest errors
with the imposed observational limits. Simulations will be 
hard coded into the script.

Must use HI_XraySoft as the field type here!!
'''




if(len(sys.argv)<4):
    print("Usage cgm_plotting_full.py 1) field_type 4) sim_set 5) sim_z 6) obs_limit 7) trial_number")    
    exit()

field_type = sys.argv[1] # Note, this is the first field to be considered. Can be a multifield (field1_field2).
sim_set = sys.argv[2] # e.g. CV
sim_z = sys.argv[3] # e.g. 0.00
if(len(sys.argv)>4):
    obs_limit = sys.argv[4] # Leave blank or '' if 6th argument.  
else:
    obs_limit = ''
if(len(sys.argv)>5):
    trial = int(sys.argv[5])
else:
    trial = -1 # This will break for plotting loss function.  

print("obs_limit= ", obs_limit)

root_dir_TNG = '/home/ng474/CMD_py/results/train_IllustrisTNG_test_IllustrisTNG'
root_dir_SIMBA = '/home/ng474/CMD_py/results/train_SIMBA_test_SIMBA'
root_dir_Astrid = '/home/ng474/CMD_py/results/train_Astrid_test_Astrid'
loss_dir_TNG = '/home/ng474/CMD_py/full_halos/full_halo_%s/losses_IllustrisTNG'%(field_type)
loss_dir_SIMBA = '/home/ng474/CMD_py/full_halos/full_halo_%s/losses_SIMBA'%(field_type)
loss_dir_Astrid = '/home/ng474/CMD_py/full_halos/full_halo_%s/losses_Astrid'%(field_type)
   
monopole1 = True
monopole2 = True
smoothing1 = 0
smoothing2 = 0
arch = 'hp3'
suffix1 = ''
if not(monopole1):  suffix1 = '%s_no_monopole'%suffix1
suffix2 = ''
if not(monopole2):  suffix2 = '%s_no_monopole'%suffix2

# BEFORE CHANGING THIS, MAKE SURE THAT ALL FILES ARE IN THE RESULTS/TRAIN_SIM_TEST_SIM DIRECTORIES
results_file_TNG = 'results_train_IllustrisTNG_%s%s_all_steps_500_500_%s_smoothing_%d%s_test_IllustrisTNG_%s%s_all_steps_500_500_%s_smoothing_%d%s_%s.txt'%(field_type,obs_limit,arch,smoothing1,suffix1,field_type,obs_limit,arch,smoothing2,suffix2,sim_set)
print("results_file_IllustrisTNG:", results_file_TNG)

results_file_SIMBA = 'results_train_SIMBA_%s%s_all_steps_500_500_%s_smoothing_%d%s_test_SIMBA_%s%s_all_steps_500_500_%s_smoothing_%d%s_%s.txt'%(field_type,obs_limit,arch,smoothing1,suffix1,field_type,obs_limit,arch,smoothing2,suffix2,sim_set)
print("results_file_SIMBA:", results_file_SIMBA)

results_file_Astrid = 'results_train_Astrid_%s%s_all_steps_500_500_%s_smoothing_%d%s_test_Astrid_%s%s_all_steps_500_500_%s_smoothing_%d%s_%s.txt'%(field_type,obs_limit,arch,smoothing1,suffix1,field_type,obs_limit,arch,smoothing2,suffix2,sim_set)
print("results_file_Astrid:", results_file_Astrid)


loss_file_TNG = 'loss_%s%s_%d_all_steps_500_500_%s_smoothing_%d.txt'%(field_type,suffix1,trial,arch,smoothing1)
loss_file_SIMBA = 'loss_%s%s_%d_all_steps_500_500_%s_smoothing_%d.txt'%(field_type,suffix1,trial,arch,smoothing1)
loss_file_Astrid = 'loss_%s%s_%d_all_steps_500_500_%s_smoothing_%d.txt'%(field_type,suffix1,trial,arch,smoothing1)
run_name_TNG = '{}_IllustrisTNG_{}_z={}{}'.format(field_type, sim_set, sim_z, obs_limit)
run_name_SIMBA = '{}_SIMBA_{}_z={}{}'.format(field_type, sim_set, sim_z, obs_limit)
run_name_Astrid = '{}_Astrid_{}_z={}{}'.format(field_type, sim_set, sim_z, obs_limit)

run_dir_TNG = root_dir_TNG
run_dir_SIMBA = root_dir_SIMBA
run_dir_Astrid = root_dir_Astrid
#run_dir = root_dir + '/Halo_' + run_name

#cm = plt.get_cmap('viridis') 
if field_type == "HI":
    cm = plt.get_cmap('magma')
if field_type == "XraySoft":
    cm = plt.get_cmap('viridis')
else:
    cm = plt.get_cmap('plasma')

nprop = 6
num_maps_TNG = sum(1 for line in open(run_dir_TNG + '/' + results_file_TNG))
num_maps_SIMBA = sum(1 for line in open(run_dir_SIMBA + '/' + results_file_SIMBA))
num_maps_Astrid = sum(1 for line in open(run_dir_Astrid + '/' + results_file_Astrid))
orientations = 8  # number of orientations- 2 flips x 4 rotations
splits     = 3   #number of axes per map
num_maps_plot_TNG = int(num_maps_TNG/orientations/splits)
num_maps_plot_SIMBA = int(num_maps_SIMBA/orientations/splits)
num_maps_plot_Astrid = int(num_maps_Astrid/orientations/splits)

params_true_TNG = np.zeros((num_maps_plot_TNG*splits,6), dtype=np.float32)
params_NN_TNG   = np.zeros((num_maps_plot_TNG*splits,6), dtype=np.float32)
errors_NN_TNG   = np.zeros((num_maps_plot_TNG*splits,6), dtype=np.float32)

params_true_SIMBA = np.zeros((num_maps_plot_SIMBA*splits,6), dtype=np.float32)
params_NN_SIMBA   = np.zeros((num_maps_plot_SIMBA*splits,6), dtype=np.float32)
errors_NN_SIMBA   = np.zeros((num_maps_plot_SIMBA*splits,6), dtype=np.float32)

params_true_Astrid = np.zeros((num_maps_plot_Astrid*splits,6), dtype=np.float32)
params_NN_Astrid   = np.zeros((num_maps_plot_Astrid*splits,6), dtype=np.float32)
errors_NN_Astrid   = np.zeros((num_maps_plot_Astrid*splits,6), dtype=np.float32)

minimum = np.array([11.5, 8.0, 0.0, 3.9, -3.6, 0.00])
maximum = np.array([14.3, 12.5, 1.0, 7.6, -1.3, 0.23])
field_renorm = [0,0,0,0,-1.87,0] # Renorm, for Solar abundances assuming Asplund+ 2009 log[Z=0.0134], can switch to K to keV if you want.  
property = ['Mhalo','M_cgm', 'fcool', 'logT', 'logZ', 'fcgm']
field_label = [r'$\log(M_{\rm{halo}})$', r'$\log(\rm{M_{cgm}})$', r'$f_{\rm{cool}}$', r'$\log(\rm{T})$', r'$\log(\rm{Z})$', r'$f_{\rm{cgm}}$']
error_label = [r'$\log(M_{\rm{halo}})$', r'$\log(\rm{M_{cgm}})$', r'$f_{\rm{cool}}$', r'$\log(\rm{T})$', r'$\log(\rm{Z})$', r'$f_{\rm{cgm}}/(\Omega_{\rm{b}}/\Omega_{\rm{M}})$']
error_label_new = [r'$\log(M_{\rm{halo}})$', r'$f_{\rm{cgm}}$', r'$\log(\rm{Z_{cgm}})$', r'$\log(\rm{M_{cgm}})$', r'$f_{\rm{cool}}$', r'$\log(\rm{T_{cgm}})$']
unit_label = [r'[${\rm M}_{\odot}$]', '[${\rm M}_{\odot}$]', '', r'[K]', r'[${\rm Z}_{\odot}$]', '']
error_renorm = np.array([1.,1.,1.,1.,1.,0.16])

#print('run_dir', run_dir)
params_true_TNG[:,0],params_NN_TNG[:,0],errors_NN_TNG[:,0] = np.loadtxt(run_dir_TNG + '/' + results_file_TNG,usecols=(0,6,12),unpack=True,max_rows=num_maps_plot_TNG*splits) # Read first 1/orientations of maps.  
params_true_TNG[:,0] -= field_renorm[0] 

params_true_SIMBA[:,0],params_NN_SIMBA[:,0],errors_NN_SIMBA[:,0] = np.loadtxt(run_dir_SIMBA + '/' + results_file_SIMBA,usecols=(0,6,12),unpack=True,max_rows=num_maps_plot_SIMBA*splits) # Read first 1/orientations of maps.  
params_true_SIMBA[:,0] -= field_renorm[0]

params_true_Astrid[:,0],params_NN_Astrid[:,0],errors_NN_Astrid[:,0] = np.loadtxt(run_dir_Astrid + '/' + results_file_Astrid,usecols=(0,6,12),unpack=True,max_rows=num_maps_plot_Astrid*splits) # Read first 1/orientations of maps.  
params_true_Astrid[:,0] -= field_renorm[0]

# np.loadtxt('{}_{}_dataset.txt'.format(field_type, simulation),usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17),unpack=True

print("len(params_true_TNG[:,0])= ", len(params_true_TNG[:,0]))
order_for_params = [0, 5, 4, 1, 2, 3]
indexes_TNG = np.arange(num_maps_plot_TNG)*splits
indexes_SIMBA = np.arange(num_maps_plot_SIMBA)*splits
indexes_Astrid = np.arange(num_maps_plot_Astrid)*splits

for i in order_for_params:
    print("i= ", i)
    params_true_TNG[:,i],params_NN_TNG[:,i],errors_NN_TNG[:,i] = np.loadtxt(run_dir_TNG + '/' + results_file_TNG,usecols=(0+i,6+i,12+i),unpack=True,max_rows=num_maps_plot_TNG*splits)
    params_true_SIMBA[:,i],params_NN_SIMBA[:,i],errors_NN_SIMBA[:,i] = np.loadtxt(run_dir_SIMBA + '/' + results_file_SIMBA,usecols=(0+i,6+i,12+i),unpack=True,max_rows=num_maps_plot_SIMBA*splits)
    params_true_Astrid[:,i],params_NN_Astrid[:,i],errors_NN_Astrid[:,i] = np.loadtxt(run_dir_Astrid + '/' + results_file_Astrid,usecols=(0+i,6+i,12+i),unpack=True,max_rows=num_maps_plot_Astrid*splits)

    params_true_TNG[:,i] -= field_renorm[i]
    params_true_SIMBA[:,i] -= field_renorm[i]
    params_true_Astrid[:,i] -= field_renorm[i]
    params_NN_TNG[:,i] -= field_renorm[i]
    params_NN_SIMBA[:,i] -= field_renorm[i]
    params_NN_Astrid[:,i] -= field_renorm[i]

    mhalo_array_TNG = np.asarray(params_true_TNG[indexes_TNG,0])
    mhalo_array_SIMBA = np.asarray(params_true_SIMBA[indexes_SIMBA,0])
    mhalo_array_Astrid = np.asarray(params_true_Astrid[indexes_Astrid,0])

# ==== Error By Halo Type Values ====

idx_subL_TNG = np.where(params_true_TNG[:,0]<12.0)
idx_L_TNG = np.where((params_true_TNG[:,0]>=12.0) & (params_true_TNG[:,0]<13.0))
idx_group_TNG = np.where(params_true_TNG[:,0]>=13.0)

idx_subL_SIMBA = np.where(params_true_SIMBA[:,0]<12.0)
idx_L_SIMBA = np.where((params_true_SIMBA[:,0]>=12.0) & (params_true_SIMBA[:,0]<13.0))
idx_group_SIMBA = np.where(params_true_SIMBA[:,0]>=13.0)

idx_subL_Astrid = np.where(params_true_Astrid[:,0]<12.0)
idx_L_Astrid = np.where((params_true_Astrid[:,0]>=12.0) & (params_true_Astrid[:,0]<13.0))
idx_group_Astrid = np.where(params_true_Astrid[:,0]>=13.0)

err_subL_TNG = np.sqrt(np.mean((params_true_TNG[idx_subL_TNG][:] - params_NN_TNG[idx_subL_TNG][:])**2, axis=0))
err_L_TNG = np.sqrt(np.mean((params_true_TNG[idx_L_TNG][:] - params_NN_TNG[idx_L_TNG][:])**2, axis=0))
err_group_TNG = np.sqrt(np.mean((params_true_TNG[idx_group_TNG][:] - params_NN_TNG[idx_group_TNG][:])**2, axis=0))

err_subL_SIMBA = np.sqrt(np.mean((params_true_SIMBA[idx_subL_SIMBA][:] - params_NN_SIMBA[idx_subL_SIMBA][:])**2, axis=0))
err_L_SIMBA = np.sqrt(np.mean((params_true_SIMBA[idx_L_SIMBA][:] - params_NN_SIMBA[idx_L_SIMBA][:])**2, axis=0))
err_group_SIMBA = np.sqrt(np.mean((params_true_SIMBA[idx_group_SIMBA][:] - params_NN_SIMBA[idx_group_SIMBA][:])**2, axis=0))

err_subL_Astrid = np.sqrt(np.mean((params_true_Astrid[idx_subL_Astrid][:] - params_NN_Astrid[idx_subL_Astrid][:])**2, axis=0))
err_L_Astrid = np.sqrt(np.mean((params_true_Astrid[idx_L_Astrid][:] - params_NN_Astrid[idx_L_Astrid][:])**2, axis=0))
err_group_Astrid = np.sqrt(np.mean((params_true_Astrid[idx_group_Astrid][:] - params_NN_Astrid[idx_group_Astrid][:])**2, axis=0))

# ==== Error By Halo Type Figure ====

x = np.arange(nprop)
fig=plt.figure(figsize=(7.5,7.5))
ax = fig.add_subplot(311)
# for i in range(nparams):
 
for i,j in enumerate(order_for_params):
    ax.bar(i+1-0.2,err_subL_TNG[j]/error_renorm[j],0.2, color='hotpink')
    ax.bar(i+1,err_subL_SIMBA[j]/error_renorm[j],0.2, color='darkviolet')
    ax.bar(i+1+0.2,err_subL_Astrid[j]/error_renorm[j],0.2, color='mediumblue')

#for i, j in enumerate(order_for_params):
#    ax.text(i + 1 - 0.2, err_subL_TNG[j] / error_renorm[j] + 0.02, f'{err_subL_TNG[j]/error_renorm[j]:.3f}', ha='center')
#    ax.text(i + 1, err_subL_SIMBA[j] / error_renorm[j] + 0.02, f'{err_subL_SIMBA[j]/error_renorm[j]:.3f}', ha='center')
#    ax.text(i + 1 + 0.2, err_subL_Astrid[j] / error_renorm[j] + 0.02, f'{err_subL_Astrid[j]/error_renorm[j]:.3f}', ha='center')
ax.axvline(x=1+1+0.5, ls='--', color='gray', label='_nolegend_')
ax.text(2.35,0.5, r'$\rm{R_{\rm{200c}}}$', rotation=90, va='center', ha='center', fontsize=15)
ax.text(2.65,0.5, r'$200\ \rm{kpc}$', rotation=90, va='center', ha='center', fontsize=15)
ax.tick_params(axis="x", labelbottom=False)
ax.tick_params(axis='both', direction='in')
#ax.set_title(r'\rm{Sub\ - L*}')
ax.set_ylabel(r'$\rm{Error}$',fontsize=18)
ax.set_ylim(0,0.6)
ax.legend([r'$\rm{IllustrisTNG}$',r'$\rm{SIMBA}$', r'$\rm{Astrid}$'],fontsize=13, loc='upper right')
ax.set_ylim(0,0.7)
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
#ax.legend([r'$\rm{Xray}$',r'$\rm{HI}$', r'$\rm{HI}+\rm{XRay}$'],fontsize=13, loc='upper right')
#if obs_limit == '':
#    ax.text(0.55, 0.6, r'\rm{Sub\ L*}',fontsize=15)
#else:
#    ax.text(0.55, 0.6, r'\rm{Sub\ L*\ .obslimit}', fontsize=15)

ax.text(0.55,0.6,r'\rm{Sub\ L*}',fontsize=15)

ax = fig.add_subplot(312)
for i,j in enumerate(order_for_params):
    ax.bar(i+1-0.2,err_L_TNG[j]/error_renorm[j],0.2, color='hotpink')
    ax.bar(i+1,err_L_SIMBA[j]/error_renorm[j],0.2, color='darkviolet')
    ax.bar(i+1+0.2,err_L_Astrid[j]/error_renorm[j],0.2, color='mediumblue')

#for i, j in enumerate(order_for_params):
#    ax.text(i + 1 - 0.2, err_L_TNG[j] / error_renorm[j] + 0.02, f'{err_L_TNG[j]/error_renorm[j]:.3f}', ha='center')
#    ax.text(i + 1, err_L_SIMBA[j] / error_renorm[j] + 0.02, f'{err_L_SIMBA[j]/error_renorm[j]:.3f}', ha='center')
#    ax.text(i + 1 + 0.2, err_L_Astrid[j] / error_renorm[j] + 0.02, f'{err_L_Astrid[j]/error_renorm[j]:.3f}', ha='center')
ax.axvline(x=1+1+0.5, ls='--', color='gray', label='_nolegend_')
ax.tick_params(axis="x", labelbottom=False)
ax.tick_params(axis='both', direction='in')
#ax.set_title(r'$\rm{L*}$')
ax.set_ylabel(r'$\rm{Error}$',fontsize=18)
ax.set_ylim(0,0.6)
ax.legend([r'$\rm{IllustrisTNG}$',r'$\rm{SIMBA}$', r'$\rm{Astrid}$'],fontsize=13, loc='upper right')
#ax.text(0.5,0.5,r'\rm{L*\ .obslimits} ',fontsize=15)
ax.set_ylim(0,0.7)
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6])
#ax.legend([r'$\rm{Xray}$',r'$\rm{HI}$', r'$\rm{HI}+\rm{XRay}$'],fontsize=13, loc='upper right')
#if obs_limit == '':
#    ax.text(0.55, 0.6, r'\rm{L*}',fontsize=15)
#else:
#    ax.text(0.55, 0.6, r'\rm{L*\ .obslimit}', fontsize=15)
ax.text(0.55,  0.6, r'\rm{L*}', fontsize=15)

ax = fig.add_subplot(313)
for i, j in enumerate(order_for_params):
    ax.bar(i+1-0.2,err_group_TNG[j]/error_renorm[j],0.2, color='hotpink')
    ax.bar(i+1,err_group_SIMBA[j]/error_renorm[j],0.2, color='darkviolet')
    ax.bar(i+1+0.2,err_group_Astrid[j]/error_renorm[j],0.2, color='mediumblue')

#for i, j in enumerate(order_for_params):
#    ax.text(i + 1 - 0.2, err_group_TNG[j] / error_renorm[j] + 0.02, f'{err_group_TNG[j]/error_renorm[j]:.3f}', ha='center')
#    ax.text(i + 1, err_group_SIMBA[j] / error_renorm[j] + 0.02, f'{err_group_SIMBA[j]/error_renorm[j]:.3f}', ha='center')
#    ax.text(i + 1 + 0.2, err_group_Astrid[j] / error_renorm[j] + 0.02, f'{err_group_Astrid[j]/error_renorm[j]:.3f}', ha='center')
ax.axvline(x=1+1+0.5, ls='--', color='gray', label='_nolegend_')
#ax.text(0.5,0.5,r'\rm{Group\ .obslimits}',fontsize=15)
ax.tick_params(axis='both', direction='in')
ax.set_xticks(x+1,error_label_new,fontsize=13)
ax.set_ylabel(r'$\rm{Error}$',fontsize=18)
ax.set_ylim(0,0.6)
ax.legend([r'$\rm{IllustrisTNG}$',r'$\rm{SIMBA}$',r'$\rm{Astrid}$'],fontsize=13, loc='upper right')
ax.set_ylim(0,0.7)
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6])
#ax.legend([r'$\rm{Xray}$',r'$\rm{HI}$', r'$\rm{HI}+\rm{XRay}$'],fontsize=13, loc='upper right')

#if obs_limit == '':
#    ax.text(0.55, 0.6, r'\rm{Group}',fontsize=15)
#else:
#    ax.text(0.55, 0.6, r'\rm{Group\ .obslimit}', fontsize=15)

ax.text(0.55, 0.6, r'\rm{Group}', fontsize=15)

fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('ErrorbySimType_{}_{}_{}.trial_{}.png'.format(field_type, sim_set, sim_z,trial), bbox_inches='tight')



# ==== end of file ====






