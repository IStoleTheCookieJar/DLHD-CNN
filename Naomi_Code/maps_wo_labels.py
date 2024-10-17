import numpy as np
import matplotlib.pyplot as plt
import sys
from pylab import get_cmap
import matplotlib as mpl
mpl.rc('figure', max_open_warning = 0)

field = sys.argv[1]
sim = sys.argv[2]
#slice_lo = int(sys.argv[2])
#slice_hi = int(sys.argv[3])

if(len(sys.argv)>3):
    obslimit = True
else:
    obslimit = False

obslimit_suffix = ''
if(obslimit):
    obslimit_suffix = '.obslimit'

cut = "halocentric_CV"
#cut = "halocentric_TNG300"

my_halo_basedir = '/home/bo256/project/halos/3_axis/'
my_CMD_basedir = '/home/bo256/project/CMD/'
#my_CMD_basedir = 'home/ng474/CMD_py/'

plot_glossy = True
plot_labels = True #False

colorbar = "viridis"

if(cut == "halocentric_CV"):
    sim = sim
    set = "CV"
    redshift = "0.00"
    simID_lo = 0
    simID_hi = 26
    haloID_max = 2 #NG: was 1000
    maxstack = (simID_hi+1-simID_lo)*haloID_max
    param_file_name = '%s/Params_%s_%s_z=%s.ascii'%(my_halo_basedir,sim,set,redshift) 
    fparam = open(param_file_name, "r")
    Mhalo, Param1, Param2, Param3, Param4, Param5 = np.loadtxt(fparam, usecols=(0,1,2,3,4,5), unpack=True)


my_field_name = field
halo_conversion = 1
if(field=="ne"):
    vmin = -7
    vmax = -1
    halo_conversion = 1/(3.086e+24)/0.7
    my_field_name = "elec"
if(field=="O_VII_abs"):
    vmin = 10
    vmax = 15
    label = "O\;VII Column" 
if(field=="Mg_II"):
    vmin = 10
    vmax = 15
if((field=="HI") | (field=="H_I")):
    colorbar = "viridis"
    vmin = 12
    vmax = 21
    label = "H I" 
    if(obslimit):
        vmin = 19
        vmax = 21
        label = "H I obslimit" 
if("Xray" in field):
    vmin = -1
    vmax = 2
if(field=="Xray"): #Deprecated  # add 42.98 to get to erg/s/kpc^2, eROSITA limit 37.3
    vmin = -8.7 #-5.68 # -8  -5.68 is eROSITA limit
    vmax = -5.7 #-2.68
if(field=="XraySoft"):
    colorbar = "magma"
    vmin = 0
    vmax = 4
    label = "X-ray" 
    if(obslimit):
        vmin = np.log10(162)
        vmax = 4
        label = "X-ray obslimit" 

if("rat" in field):
    vmin = -0.5
    vmax = 0.5
if(field=="star"):
    colorbar = "cividis"
    vmin = -5
    vmax = 0
    label = "Stellar Mass" 

mid_lin = 0 # for Group plots
if((field=="O_VII")):
    mid_lin = 3.13
    label = "O VII" 
if((field=="O_VIII")):
    mid_lin = 2.40
    label = "O VIII" 
if((field=="Fe_XVII")):
    mid_lin = 2.14
    label = "Fe XVII" 

#vmin = 0
#vmax = 2

if((field=="0.4_0.6keV_lem")): 
    mid_lin = 47.5
    label = "0.4-0.6 keV" 
if((field=="0.73_1.1keV_lem")): 
    mid_lin = 69.0
    label = "0.73-1.1 keV" 
if((field=="1.43_2.0keV_lem")):
    mid_lin = 14.7
    label = "1.43-2.0 keV" 

if(mid_lin > 0 ):
    vmin = np.log10(mid_lin)-1.5
    vmax = np.log10(mid_lin)+1.5


#vmin = 0.5
#vmax = 2.5



# Plot stack is to go into the stack and 
plot_stack = True
plot_frame = False

if(plot_stack):
    halo_stack = np.load(my_halo_basedir + 'Maps_%s_%s_%s_z=0.00.npy'%(field,sim,set))

    #print("halo_stack.shape= ", halo_stack.shape)

for i in range(maxstack):
#    if(i<1000): continue
    if(cut=="CMD"):
        simID = int(i/15)
        axis = int(i/5)-simID*3
        slice = int(i%5)
        #print(i,simID,axis,slice)        
    if("halocentric" in cut):
        axis = i%3 #2
        simID = int((i/3)/haloID_max)+simID_lo
        haloID = int(int(i/3)%haloID_max)
        #print(i,simID,haloID,axis,Mhalo[haloID])        

    if(plot_stack):
        fig = plt.figure(figsize=(2.5,2.5))
        #ax = fig.add_subplot(111)
        ax = plt.Axes(fig, [0., 0., 1, 1])

        
        ax_obj = ax.imshow(np.log10(halo_stack[i]),vmin=vmin,vmax=vmax,cmap=get_cmap(colorbar))

        if(plot_glossy):
            ax.set_axis_off()
            fig.add_axes(ax)

            plot_halo_mass_top = False
            if(plot_halo_mass_top):
                if(plot_labels):
                    ax.text(64, 4, '',color="white", fontsize=18, horizontalalignment='center', verticalalignment='top',weight='bold')
                    ax.text(64, 4, '',color="white", fontsize=18, horizontalalignment='center', verticalalignment='top',weight='bold')
                    ax.text(4, 64, '',color="white", rotation="vertical", fontsize=18, horizontalalignment='left', verticalalignment='center',weight='bold')
            else:
                if(plot_labels):
                    ax.text(64, 4, '',color="white", fontsize=14, horizontalalignment='center', verticalalignment='top',weight='bold')
                    ax.text(4, 64, '',color="white", rotation="vertical", fontsize=18, horizontalalignment='left', verticalalignment='center',weight='bold')
                    ax.text(4, 64, '',color="white", rotation="vertical", fontsize=14, horizontalalignment='left', verticalalignment='center',weight='bold')
                    ax.text(64, 128, '', color="white", fontsize=14, horizontalalignment='center', verticalalignment='bottom', weight='bold')


        if(not plot_glossy):
            cax = plt.axes([0.88, 0.1, 0.03, 0.8])
            fig.colorbar(ax_obj, cax=cax)
        
        mean = np.mean(halo_stack[i])
        min = np.min(halo_stack[i])
        #print("mean,min= ", mean,min)

        if(cut=="CMD"):
            if(not plot_glossy):
                ax.text(150,-20,"Mean= " + str(mean))
                ax.text(0,-20,"CMD_%s[%d]"%(field,i))
            fig.savefig("halo_maps/CMD_%s.simID_%d.slice_%d.axis_%d%s.png"%(field,simID,slice,axis,obslimit_suffix))
        if("halocentric" in cut):
            if(not plot_glossy):
                ax.text(75,-10,"Mean= " + str(mean))
                ax.text(0,-10,"halo_%s[%d]"%(field,i))
            if(sim=="TNG300"):
                fig.savefig("halo_maps/Group_%s.simID_%d.halo_%d.axis_%d%s.png"%(field,simID,haloID,axis,obslimit_suffix))
            else:
                fig.savefig("halo_maps/halo_%s.simID_%d.halo_%d.axis_%d%s.png"%(field,simID,haloID,axis,obslimit_suffix))


    if(plot_frame):
        if(cut=="CMD"):
            CMD_field = np.load(my_CMD_basedir + '/%s/LH_%d/snap_033/%s.LH_%d.snap_033.slice_%d.axis_%d.%s.npy'%(sim,simID,sim,simID,slice,axis,my_field_name))
        if("halocentric" in cut):
            halo_field = np.load(my_halo_basedir + '/%s/%s_%d/snap_033/%s.%s_%d.snap_033.halo_%d.axis_%d.%s.npy'%(sim,set,simID,sim,set,simID,haloID,axis,my_field_name))

        #print("halo_field.shape= ", halo_field.shape)
    
        fig = plt.figure(figsize=(2.5,2.5))
        ax = fig.add_subplot(111)
        
        halo_field *= halo_conversion  
        halo_field = np.swapaxes(halo_field,0,1)

        ax_obj = ax.imshow(np.log10(halo_field),vmin=vmin,vmax=vmax)

        cax = plt.axes([0.88, 0.1, 0.03, 0.8])
        fig.colorbar(ax_obj, cax=cax)

        mean = np.mean(halo_field)
        if(cut=="CMD"):
            ax.text(0,-20,"CMD_%s simID_%d.slice_%d.axis_%d"%(field,simID,slice,axis))
            ax.text(0,-10,"ID_%d"%(i))
            ax.text(150,-10,"Mean= " + str(mean))        
        else:
            ax.text(0,-10,"halo_%s simID_%d.haloID_%d.axis_%d"%(field,simID,haloID,axis))
            ax.text(0,-5,"ID_%d"%(i))
            ax.text(75,-5,"Mean= " + str(mean))
        

        if(cut=="CMD"):
            fig.savefig("halo_maps/CMD_%s.simID_%d.slice_%d.axis_%d.png"%(field,simID,slice,axis))
        if("halocentric" in cut):
            fig.savefig("halo_maps/%s/halo_%s_%s.simID_%d.halo_%d.axis_%d.png"%(sim,field,sim,simID,haloID,axis))
