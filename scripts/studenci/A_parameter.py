import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, gamma,beta

def check_symmetry(n,mean,std,x):
    factor = n/(n-1)/(n-2)
    standarisation = np.sum(((x - mean) / std)**3)
    A = factor * standarisation
    return A
    

np.random.seed(42)

n = 1000 

x_sym = norm.rvs(size=n,loc=50,scale=10)

x_pos = gamma.rvs(size=n,a=2,loc=0,scale=10)

x_temp = gamma.rvs(size=n, a=2, loc=0, scale=10)

x_neg = -x_temp + 100  



data_sets = [
    (x_sym, "Symetryczny - normalny"),
    (x_pos, "Asymetryczny - prawoskośny"),
    (x_neg, "Asymetryczny - lewoskośny")
]

A_values = []
for dist,title in data_sets:
    mean = np.mean(dist)
    std = np.std(dist, ddof=1)
    A = check_symmetry(n,mean,std,dist)
    A_values.append(A)
    print(f"Rozkład {title}: A = {A:.4f}")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (ax, (data, title)) in enumerate(zip(axes, data_sets)):
    ax.hist(data, bins=50, color="blue", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Wartość')
    ax.set_ylabel('Częstość')
    
    # Dodaj tekst z wartością A na wykres
    ax.text(0.05, 0.95, f'A = {A_values[i]:.4f}', 
            transform=ax.transAxes, 
            fontsize=12, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.show()

for dist,title in data_sets:
    mean = np.mean(dist)
    std = np.std(dist, ddof=1)
    A = check_symmetry(n,mean,std,dist)
    print(f"Rozkład {title}: A = {A:.4f}")