from class_microstates import Microstates
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.axes import Axes
from mne import create_info, Info
from mne.channels.layout import _find_topomap_coords
from mne.viz import plot_topomap 



def visualise_base_microstates(base_microstates:np.ndarray,figsize:tuple[int,int],info:object,show:bool=True):
    n_microstates = base_microstates.shape[0] #pobranie liczby mikrostanów
    microstates_ax_labels = list(range(1,n_microstates+1)) #utworzenie etykiet dla subplotów
    fig,axes = plt.subplots(1,n_microstates,figsize=figsize) #utworzenie figury i subplotów
    for m, (microstate, micro_ax_label) in enumerate(zip(base_microstates,microstates_ax_labels)):
        ax = axes[m]
        plot_topomap(microstate, pos=info, axes=ax,show=False)
        ax.set_title(f"Mikrostan {micro_ax_label}")
    
    if show:
        plt.show()
    
    return fig,axes

