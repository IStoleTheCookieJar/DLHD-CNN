
'''
Use this function to trim the datasets so that
they have less points and the error bars are more 
clearly visible.
'''

def trim_mhalo_data(logM,sim):

    '''
    Usage in code: just before ax.errorbar line,
    if(j%trim_mhalo_data(params_true[index,0]) != 0): continue
    '''
    if(sim=="300"):
        logM_plot_all = 14.0 # Is actually 2x less than this.  
    if(sim=="100"):
        logM_plot_all = 13.0
    if(logM>=logM_plot_all):
        n = 1
    else:
        n = int(10**logM_plot_all/10**logM)

    if(n<1):
        n=1

    return(n)




