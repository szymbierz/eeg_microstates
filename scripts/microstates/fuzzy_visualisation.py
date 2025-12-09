from scripts.microstates.visualise_microstates import visualise_base_microstates
import numpy as np
import matplotlib.pyplot as plt 

def visualise_fuzzy_microstates(base_microstates:np.ndarray, U:np.ndarray, 
                              figsize:tuple[int,int], info:object, show:bool=True):
    n_microstates = base_microstates.shape[0]
    

    fig1, axes1 = visualise_base_microstates(base_microstates, figsize, info, show=False)

    fig2, axes2 = plt.subplots(1, n_microstates, figsize=figsize)
    for m in range(n_microstates):
        ax = axes2[m]
        ax.plot(U[m,:])
        ax.set_title(f"Przynależność do mikrostanu {m+1}")
    
    if show:
        plt.show()
    
    return fig1, fig2, axes1, axes2