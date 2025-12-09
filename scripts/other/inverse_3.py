import mne 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
# Ustawienia podstawowe 
raw_path = "/Users/szymbierz/Desktop/wszystko/notebooks/statystyka/inne/eeg/eeg_files/20241017_kp_cleaned.edf"
ica =mne.preprocessing.read_ica('best_ica_19.05-ica.fif')
raw = mne.io.read_raw_edf(raw_path, preload=True, exclude=["EXG1","EXG2","EXG3","EXG4","EXG5"])
raw.set_montage("biosemi64")
raw.crop(0, 200)  
raw.filter(8, 40)
raw_copy = raw.copy()
ica.apply(raw_copy)

# Model sfery głowy
sphere_model = mne.make_sphere_model(info=raw.info)

src = mne.setup_volume_source_space(subject=None, pos=5.0) # 8mm spacing
print(f"Liczba dipoli: {src[0]['nuse']}")

# Forward solution
fwd = mne.make_forward_solution(raw.info, trans=None, src=src, 
                               bem=sphere_model, eeg=True, meg=False)
fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)

# Macierz zysków (Lead field matrix)
G = fwd_fixed['sol']['data']  # kształt: (n_kanałów, n_źródeł)
print(f"Wymiary macierzy zysków: {G.shape}")

# ROZWIĄZANIE PROBLEMU ODWROTNEGO METODĄ MONTE CARLO


epochs = mne.make_fixed_length_epochs(raw_copy, duration=1.0)
evoked = epochs.average()
M = evoked.data[:, 200]  
print(f"Pomiary z elektrod: {M.shape}")


pos = fwd_fixed['source_rr']  
n_dipoles = pos.shape[0]
n_channels = M.shape[0]


n_iterations = 500
n_active = 8
best_error = np.inf
best_dipoles = None
best_amplitudes = None
best_error_history = []
error_history = []
# Normalizacja danych
M_norm = M / np.linalg.norm(M) #dla zachowania jednakowej skali
print("\nRozwiązywanie problemu odwrotnego metodą Monte Carlo...")
for i in range(n_iterations):
        active_dipoles = np.random.choice(n_dipoles, n_active, replace=False)
        G_subset = G[:, active_dipoles]
        lambda_param = 0.1
        GTG = G_subset.T @ G_subset + lambda_param * np.eye(n_active)
        GTM = G_subset.T @ M_norm
        amplitudes = np.linalg.solve(GTG, GTM)
        M_reconstructed = G_subset @ amplitudes
        error = np.linalg.norm(M_norm - M_reconstructed)
        error_history.append(error)
        if error < best_error:
            best_error = error
            best_dipoles = active_dipoles
            best_amplitudes = amplitudes       
        if i % 100 == 0:
            print(f"Iteracja {i}, najlepszy błąd: {best_error:.4f}")
            best_error_history.append(best_error)
print(f"\nNajlepszy błąd rekonstrukcji: {best_error:.4f}")
print(f"Liczba znalezionych aktywnych dipoli: {len(best_dipoles)}")

# WIZUALIZACJA 3D AKTYWNOŚCI DIPOLI
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# DODAJ SIATKĘ SFERYCZNĄ REPREZENTUJĄCĄ MÓZG
# Parametry sfery
radius = 0.095  # promień głowy w metrach (95mm)
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
x_sphere = radius * np.outer(np.cos(u), np.sin(v))
y_sphere = radius * np.outer(np.sin(u), np.sin(v))
z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

#siatka
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, 
                  color='gray', alpha=0.5, linewidth=0.5)


active_pos = pos[best_dipoles]
colors = plt.cm.hot(np.abs(best_amplitudes) / np.max(np.abs(best_amplitudes)))
sizes = 100 * np.abs(best_amplitudes) / np.max(np.abs(best_amplitudes))

scatter = ax.scatter(active_pos[:, 0], active_pos[:, 1], active_pos[:, 2],
                    c=colors, s=sizes, alpha=0.8, 
                    edgecolors='black', linewidth=0.5,
                    label='Aktywne dipole')


mappable = plt.cm.ScalarMappable(cmap='hot')
mappable.set_array(np.abs(best_amplitudes))
cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label('Amplituda dipola', rotation=270, labelpad=20)

# Ustawienia wykresu
ax.set_xlabel('X [m]', fontsize=10)
ax.set_ylabel('Y [m]', fontsize=10)
ax.set_zlabel('Z [m]', fontsize=10)
ax.set_title('Rekonstrukcja źródeł aktywności mózgu\nmetodą Monte Carlo', fontsize=16, pad=20)

# proporcje osi 
ax.set_box_aspect([1,1,1])

#  limity osi
ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-0.1, 0.1])
ax.set_zlim([-0.1, 0.1])

# Dodaj siatkę i ustaw widok
ax.grid(True, alpha=0.3)
ax.view_init(elev=25, azim=45)

# Legenda
ax.legend(loc='upper left', fontsize=10)

# opisy anatomiczne

ax.text(0, 0.12, 0, 'A', fontsize=12, weight='bold')  # Przód (Anterior)
ax.text(0, -0.12, 0, 'P', fontsize=12, weight='bold')  # Tył (Posterior)
ax.text(0, 0, 0.12, 'S', fontsize=12, weight='bold')  # Góra (Superior)

plt.tight_layout()

epochs.plot_psd_topomap(ch_type="eeg")

# wykres zbieżności błędu
plt.figure("Zbieżność błędu Monte Carlo")
plt.plot(range(0, n_iterations, 100), best_error_history, marker='o', linestyle='-')
plt.grid(True)
plt.title('Zbieżność błędu w kolejnych iteracjach')
plt.xlabel('Iteracja')
plt.ylabel('Najlepszy zanotowany błąd')

# WIZUALIZACJA KORELACJI  

G_best_subset = G[:, best_dipoles]
M_reconstructed_best = G_best_subset @ best_amplitudes


correlation = np.corrcoef(M_norm, M_reconstructed_best)[0, 1]


plt.figure("Jakość dopasowania modelu", figsize=(8, 8))
plt.scatter(M_norm, M_reconstructed_best, alpha=0.7, edgecolors='k', label='Kanały EEG')


lims = [
    np.min([plt.xlim(), plt.ylim()]),
    np.max([plt.xlim(), plt.ylim()]),
]
plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Idealne dopasowanie')

# Ustawienia wykresu
plt.grid(True)
plt.title(f'Dopasowanie modelu do danych\nWspółczynnik korelacji: {correlation:.4f}', fontsize=14)
plt.xlabel('Oryginalne dane (znormalizowane)', fontsize=12)
plt.ylabel('Dane zrekonstruowane przez model', fontsize=12)
plt.legend()
plt.axis('equal')
plt.tight_layout()


M_reconstructed_best = G[:, best_dipoles] @ best_amplitudes


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
vmax = np.max(np.abs(M_norm))
vmin = -vmax


mne.viz.plot_topomap(M_norm, raw.info, axes=ax1, show=False, vlim=(vmin, vmax))
ax1.set_title('Oryginalne dane')


mne.viz.plot_topomap(M_reconstructed_best, raw.info, axes=ax2, show=False, vlim=(vmin, vmax))
ax2.set_title('Dane zrekonstruowane')


residuals = M_norm - M_reconstructed_best
mne.viz.plot_topomap(residuals, raw.info, axes=ax3, show=False, vlim=(vmin, vmax))
ax3.set_title('Błąd (Reszty)')

plt.tight_layout()

plt.show()



#statystyki
print(f"\nStatystyki aktywnych dipoli:")
print(f"Średnia amplituda: {np.mean(np.abs(best_amplitudes)):.4f}")
print(f"Max amplituda: {np.max(np.abs(best_amplitudes)):.4f}")
print(f"Min amplituda: {np.min(np.abs(best_amplitudes)):.4f}")


