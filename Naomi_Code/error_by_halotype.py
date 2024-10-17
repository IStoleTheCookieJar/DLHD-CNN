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

NOTE: The first field that should be typed in is XraySoft.

As of 8/8/23, this is the most up-to-date version.
'''




if(len(sys.argv)<6):
    print("Usage cgm_plotting_full.py 1) field_type 2) field_type_B 3) simulation 4) sim_set 5) sim_z 6) obs_limit 7) trial_number")    
    exit()

field_type = sys.argv[1] # Note, this is the first field to be considered. Can be a multifield.
field_type_B = sys.argv[2] # Note, this is the 2nd field to be considered. Cannot be a multifield.
simulation = sys.argv[3] # e.g. IllustrisTNG
sim_set = sys.argv[4] # e.g. CV
sim_z = sys.argv[5] # e.g. 0.00
if(len(sys.argv)>6):
    obs_limit = sys.argv[6] # Leave blank or '' if 6th argument.  
else:
    obs_limit = ''
if(len(sys.argv)>7):
    trial = int(sys.argv[7])
else:
    trial = -1 # This will break for plotting loss function.  

print("obs_limit= ", obs_limit)

root_dir = '/home/ng474/CMD_py/results/train_%s_test_%s'%(simulation, simulation)
loss_dir = '/home/ng474/CMD_py/full_halos/full_halo_%s/losses_%s'%(field_type, simulation)
loss_dir_B = '/home/ng474/CMD_py/full_halos/full_halo_%s/losses_%s'%(field_type_B, simulation)
loss_dir_C = '/home/ng474/CMD_py/full_halos/full_halo_%s,%s/losses_%s'%(field_type, field_type_B, simulation)
   
monopole1 = True
monopole2 = True
smoothing1 = 0
smoothing2 = 0
arch = 'hp3'
suffix1 = ''
if not(monopole1):  suffix1 = '%s_no_monopole'%suffix1
suffix2 = ''
if not(monopole2):  suffix2 = '%s_no_monopole'%suffix2

results_file = 'results_train_%s_%s%s_all_steps_500_500_%s_smoothing_%d%s_test_%s_%s%s_all_steps_500_500_%s_smoothing_%d%s_%s.txt'%(simulation,field_type,obs_limit,arch,smoothing1,suffix1,simulation,field_type,obs_limit,arch,smoothing2,suffix2,sim_set)
print("results_file=", results_file)
results_file_B = 'results_train_%s_%s%s_all_steps_500_500_%s_smoothing_%d%s_test_%s_%s%s_all_steps_500_500_%s_smoothing_%d%s_%s.txt'%(simulation,field_type_B,obs_limit,arch,smoothing1,suffix1,simulation,field_type_B,obs_limit,arch,smoothing2,suffix2,sim_set)
print("results_file= ", results_file_B)

results_file_C = 'results_train_%s_%s_%s%s_all_steps_500_500_%s_smoothing_%d%s_test_%s_%s_%s%s_all_steps_500_500_%s_smoothing_%d%s_%s.txt'%(simulation,field_type_B,field_type,obs_limit,arch,smoothing1,suffix1,simulation,field_type_B,field_type,obs_limit,arch,smoothing2,suffix2,sim_set)
print("results_file= ", results_file_C)


loss_file = 'loss_%s%s_%d_all_steps_500_500_%s_smoothing_%d.txt'%(field_type,suffix1,trial,arch,smoothing1)
loss_file_B = 'loss_%s%s_%d_all_steps_500_500_%s_smoothing_%d.txt'%(field_type_B,suffix1,trial,arch,smoothing1)
loss_file_C = 'loss_%s_%s%s_%d_all_steps_500_500_%s_smoothing_%d.txt'%(field_type_B, field_type ,suffix1,trial,arch,smoothing1)
run_name = '{}_{}_{}_z={}{}'.format(field_type, simulation, sim_set, sim_z, obs_limit)
run_name_B = '{}_{}_{}_z={}{}'.format(field_type_B, simulation, sim_set, sim_z, obs_limit)
run_name_C = '{}_{}_{}_{}_z={}{}'.format(field_type_B, field_type, simulation, sim_set, sim_z, obs_limit)
run_dir = root_dir
#run_dir = root_dir + '/Halo_' + run_name

#cm = plt.get_cmap('viridis') 
if field_type == "HI":
    cm = plt.get_cmap('magma')
if field_type == "XraySoft":
    cm = plt.get_cmap('viridis')
else:
    cm = plt.get_cmap('plasma')

nprop = 6
num_maps = sum(1 for line in open(run_dir + '/' + results_file))
orientations = 8  # number of orientations- 2 flips x 4 rotations
splits     = 3   #number of axes per map
num_maps_plot = int(num_maps/orientations/splits)

params_true = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
params_NN   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
errors_NN   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)

params_true_B = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
params_NN_B   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
errors_NN_B   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)

params_true_C = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
params_NN_C   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)
errors_NN_C   = np.zeros((num_maps_plot*splits,6), dtype=np.float32)

minimum = np.array([11.5, 8.0, 0.0, 3.9, -3.6, 0.00])
maximum = np.array([14.3, 12.5, 1.0, 7.6, -1.3, 0.23])
field_renorm = [0,0,0,0,0,0] # Z=-1.87 Renorm, for Solar abundances assuming Asplund+ 2009 log[Z=0.0134], can switch to K to keV if you want.  
property = ['Mhalo','M_cgm', 'fcool', 'logT', 'logZ', 'fcgm']
field_label = [r'$\log(M_{\rm{halo}})$', r'$\log(\rm{M_{cgm}})$', r'$f_{\rm{cool}}$', r'$\log(\rm{T})$', r'$\log(\rm{Z})$', r'$f_{\rm{cgm}}$']
error_label = [r'$\log(M_{\rm{halo}})$', r'$\log(\rm{M_{cgm}})$', r'$f_{\rm{cool}}$', r'$\log(\rm{T})$', r'$\log(\rm{Z})$', r'$f_{\rm{cgm}}/(\Omega_{\rm{b}}/\Omega_{\rm{M}})$']
error_label_new = [r'$\log(M_{\rm{halo}})$', r'$f_{\rm{cgm}}$', r'$\log(\rm{Z_{\rm{cgm}}})$', r'$\log(\rm{M_{cgm}})$', r'$f_{\rm{cool}}$', r'$\log(\rm{T_{\rm{cgm}}})$']
unit_label = [r'[${\rm M}_{\odot}$]', '[${\rm M}_{\odot}$]', '', r'[K]', r'[${\rm Z}_{\odot}$]', '']
error_renorm = np.array([1.,1.,1.,1.,1.,0.16])

print('run_dir', run_dir)
params_true[:,0],params_NN[:,0],errors_NN[:,0] = np.loadtxt(run_dir + '/' + results_file,usecols=(0,6,12),unpack=True,max_rows=num_maps_plot*splits) # Read first 1/orientations of maps.  
params_true[:,0] -= field_renorm[0] 

params_true_B[:,0],params_NN_B[:,0],errors_NN_B[:,0] = np.loadtxt(run_dir + '/' + results_file_B,usecols=(0,6,12),unpack=True,max_rows=num_maps_plot*splits) # Read first 1/orientations of maps.  
params_true_B[:,0] -= field_renorm[0]

params_true_C[:,0],params_NN_C[:,0],errors_NN_C[:,0] = np.loadtxt(run_dir + '/' + results_file_C,usecols=(0,6,12),unpack=True,max_rows=num_maps_plot*splits) # Read first 1/orientations of maps.  
params_true_C[:,0] -= field_renorm[0]

# np.loadtxt('{}_{}_dataset.txt'.format(field_type, simulation),usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17),unpack=True

print("len(params_true[:,0])= ", len(params_true[:,0]))
order_for_params = [0, 5, 4, 1, 2, 3]
indexes = np.arange(num_maps_plot)*splits

for i in order_for_params:
    print("i= ", i)
    params_true[:,i],params_NN[:,i],errors_NN[:,i] = np.loadtxt(run_dir + '/' + results_file,usecols=(0+i,6+i,12+i),unpack=True,max_rows=num_maps_plot*splits)
    params_true_B[:,i],params_NN_B[:,i],errors_NN_B[:,i] = np.loadtxt(run_dir + '/' + results_file_B,usecols=(0+i,6+i,12+i),unpack=True,max_rows=num_maps_plot*splits)
    params_true_C[:,i],params_NN_C[:,i],errors_NN_C[:,i] = np.loadtxt(run_dir + '/' + results_file_C,usecols=(0+i,6+i,12+i),unpack=True,max_rows=num_maps_plot*splits)

    params_true[:,i] -= field_renorm[i]
    params_true_B[:,i] -= field_renorm[i]
    params_true_C[:,i] -= field_renorm[i]
    params_NN[:,i] -= field_renorm[i]
    params_NN_B[:,i] -= field_renorm[i]
    params_NN_C[:,i] -= field_renorm[i]

    mhalo_array = np.asarray(params_true[indexes,0])
    mhalo_array_B = np.asarray(params_true_B[indexes,0])
    mhalo_array_C = np.asarray(params_true_C[indexes,0])

    fig=plt.figure(figsize=(7.5,7.5))
    ax = fig.add_subplot(111) 

    ax.set_xlabel(r'$\rm{Truth}$',fontsize=22) # was 20
    ax.set_ylabel(r'$\rm{Inference}$',fontsize=22)
    ax.tick_params(axis='both', direction='in', which='major', labelsize=13)
    
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
        cb.ax.set_xlabel(r'$\log(M_{halo}/M_{\odot})$',labelpad=16,fontsize=16) # was 14

        
    ax.set_title(field_label[i] + " " + r'$\rm{%s}\ \rm{%s}$'%(field_type, simulation),fontsize=18)

    ax.plot([minimum[i],maximum[i]], [minimum[i],maximum[i]], color='k')
    #ax.text((maximum[i]-minimum[i])*0.03+minimum[i], (maximum[i]-minimum[i])*0.97+minimum[i],'Trial %d'%trial,fontsize=13)

    ax.set_xlim(minimum[i]-(maximum[i]-minimum[i])*0.1,maximum[i]+(maximum[i]-minimum[i])*0.1) # Unmoving boundaries at +/-10% of range.  
    ax.set_ylim(minimum[i]-(maximum[i]-minimum[i])*0.1,maximum[i]+(maximum[i]-minimum[i])*0.1) # Unmoving boundaries at +/-10% of range. 
    #fig.savefig('test_plot_{}_{}_{}_{}_z={}{}.trial_{}.png'.format(property[i],field_type, simulation, sim_set, sim_z, obs_limit,trial) )

# ==== Error By Halo Type Values ====

indexes_subL = np.where(params_true[:,0]<12.0)
indexes_L = np.where((params_true[:,0]>=12.0) & (params_true[:,0]<13.0))
indexes_Group = np.where(params_true[:,0]>=13.0)

indexes_subL_B = np.where(params_true_B[:,0]<12.0)
indexes_L_B = np.where((params_true_B[:,0]>=12.0) & (params_true_B[:,0]<13.0))
indexes_Group_B = np.where(params_true_B[:,0]>=13.0)

indexes_subL_C = np.where(params_true_C[:,0]<12.0)
indexes_L_C = np.where((params_true_C[:,0]>=12.0) & (params_true_C[:,0]<13.0))
indexes_Group_C = np.where(params_true_C[:,0]>=13.0)

error_subL = np.sqrt(np.mean((params_true[indexes_subL][:] - params_NN[indexes_subL][:])**2, axis=0))
error_L = np.sqrt(np.mean((params_true[indexes_L][:] - params_NN[indexes_L][:])**2, axis=0))
error_Group = np.sqrt(np.mean((params_true[indexes_Group][:] - params_NN[indexes_Group][:])**2, axis=0))

error_subL_B = np.sqrt(np.mean((params_true_B[indexes_subL_B][:] - params_NN_B[indexes_subL_B][:])**2, axis=0))
error_L_B = np.sqrt(np.mean((params_true_B[indexes_L_B][:] - params_NN_B[indexes_L_B][:])**2, axis=0))
error_Group_B = np.sqrt(np.mean((params_true_B[indexes_Group_B][:] - params_NN_B[indexes_Group_B][:])**2, axis=0))

error_subL_C = np.sqrt(np.mean((params_true_C[indexes_subL_C][:] - params_NN_C[indexes_subL_C][:])**2, axis=0))
error_L_C = np.sqrt(np.mean((params_true_C[indexes_L_C][:] - params_NN_C[indexes_L_C][:])**2, axis=0))
error_Group_C = np.sqrt(np.mean((params_true_C[indexes_Group_C][:] - params_NN_C[indexes_Group_C][:])**2, axis=0))

# ==== Error By Halo Type Figure ====

x = np.arange(nprop)
fig=plt.figure(figsize=(7.5,7.5))
ax = fig.add_subplot(311)
# for i in range(nparams):
 
for i,j in enumerate(order_for_params):
    ax.bar(i+1-0.2,error_subL[j]/error_renorm[j],0.2, color='darkblue')
    #ax.bar_label(ax.containers[i])
    ax.bar(i+1,error_subL_B[j]/error_renorm[j],0.2, color='blue')
    ax.bar(i+1+0.2,error_subL_C[j]/error_renorm[j],0.2, color='dodgerblue')

#for i, j in enumerate(order_for_params):
#    ax.text(i + 1 - 0.2, error_subL[j] / error_renorm[j] + 0.02, f'{error_subL[j]/error_renorm[j]:.3f}', ha='center')
#    ax.text(i + 1, error_subL_B[j] / error_renorm[j] + 0.02, f'{error_subL_B[j]/error_renorm[j]:.3f}', ha='center')
#    ax.text(i + 1 + 0.2, error_subL_C[j] / error_renorm[j] + 0.02, f'{error_subL_C[j]/error_renorm[j]:.3f}', ha='center')
ax.axvline(x=1+1+0.5, ls='--', color='gray', label='_nolegend_')
ax.text(2.35,0.5, r'$\rm{R_{\rm{200c}}}$', rotation=90, va='center', ha='center', fontsize=15)
ax.text(2.65,0.5, r'$200\ \rm{kpc}$', rotation=90, va='center', ha='center', fontsize=15)
ax.tick_params(axis="x", labelbottom=False)
ax.tick_params(axis='both', direction='in')
#ax.set_title(r'\rm{Sub\ - L*}')
ax.set_ylabel(r'$\rm{Error}$',fontsize=18)
ax.set_ylim(0,0.7)
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7])

ax.legend([r'$\rm{X–ray}$',r'$\rm{HI}$', r'$\rm{HI}+\rm{X–ray}$'],fontsize=13, loc='upper right')

#if obs_limit == '':
#    ax.text(0.55, 0.6, r'\rm{Sub\ L*}',fontsize=15)
#else:
#    ax.text(0.55, 0.6, r'\rm{Sub\ L*\ .obslimit}', fontsize=15)

ax.text(0.55,0.6,r'\rm{Sub\ L*}',fontsize=15)

ax = fig.add_subplot(312)
for i,j in enumerate(order_for_params):
    ax.bar(i+1-0.2,error_L[j]/error_renorm[j],0.2, color='g')
    ax.bar(i+1,error_L_B[j]/error_renorm[j],0.2, color='mediumseagreen')
    ax.bar(i+1+0.2,error_L_C[j]/error_renorm[j],0.2, color='mediumaquamarine')

#for i, j in enumerate(order_for_params):
#    ax.text(i + 1 - 0.2, error_L[j] / error_renorm[j] + 0.02, f'{error_L[j]/error_renorm[j]:.3f}', ha='center')
#    ax.text(i + 1, error_L_B[j] / error_renorm[j] + 0.02, f'{error_L_B[j]/error_renorm[j]:.3f}', ha='center')
#    ax.text(i + 1 + 0.2, error_L_C[j] / error_renorm[j] + 0.02, f'{error_L_C[j]/error_renorm[j]:.3f}', ha='center')
ax.axvline(x=1+1+0.5, ls='--', color='gray', label='_nolegend_')
ax.tick_params(axis="x", labelbottom=False)
ax.tick_params(axis='both', direction='in')
#ax.set_title(r'$\rm{L*}$')
ax.set_ylabel(r'$\rm{Error}$',fontsize=18)
ax.set_ylim(0,0.7)
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6])
ax.legend([r'$\rm{X–ray}$',r'$\rm{HI}$', r'$\rm{HI}+\rm{X–ray}$'],fontsize=13, loc='upper right')

#if obs_limit == '':
#    ax.text(0.55, 0.6, r'\rm{L*}',fontsize=15)
#else:
#    ax.text(0.55, 0.6, r'\rm{L*\ .obslimit}', fontsize=15)


ax.text(0.55,0.6,r'\rm{L*}',fontsize=15)

ax = fig.add_subplot(313)
for i, j in enumerate(order_for_params):
    ax.bar(i+1-0.2,error_Group[j]/error_renorm[j],0.2, color='darkgoldenrod')
    ax.bar(i+1,error_Group_B[j]/error_renorm[j],0.2, color='goldenrod')
    ax.bar(i+1+0.2,error_Group_C[j]/error_renorm[j],0.2, color='gold')

#for i, j in enumerate(order_for_params):
#    ax.text(i + 1 - 0.2, error_Group[j] / error_renorm[j] + 0.02, f'{error_Group[j]/error_renorm[j]:.3f}', ha='center')
#    ax.text(i + 1, error_Group_B[j] / error_renorm[j] + 0.02, f'{error_Group_B[j]/error_renorm[j]:.3f}', ha='center')
#    ax.text(i + 1 + 0.2, error_Group_C[j] / error_renorm[j] + 0.02, f'{error_Group_C[j]/error_renorm[j]:.3f}', ha='center')

#if obs_limit == '':
#    ax.text(0.55, 0.6, r'\rm{Group}',fontsize=15)
#else:
#    ax.text(0.55, 0.6, r'\rm{Group\ .obslimit}', fontsize=15)
ax.axvline(x=1+1+0.5, ls='--', color='gray', label='_nolegend_')
ax.text(0.55,0.6,r'\rm{Group}',fontsize=15)
ax.tick_params(axis='both', direction='in')
ax.set_xticks(x+1,error_label_new,fontsize=13)
ax.set_ylabel(r'$\rm{Error}$',fontsize=18)
ax.set_ylim(0,0.7)
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6])
ax.legend([r'$\rm{X–ray}$',r'$\rm{HI}$',r'$\rm{HI}+\rm{X–ray}$'],fontsize=13, loc='upper right')
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('ErrorbyHaloTypeNG_v3_{}.trial_{}.png'.format(run_name,trial), bbox_inches='tight')



# ==== end of file ====






