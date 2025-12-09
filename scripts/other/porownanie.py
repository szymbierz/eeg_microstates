import mne
import numpy as np
import matplotlib.pyplot as plt
import time

# Wczytanie EEG
raw_path = "/Users/szymbierz/Desktop/notebooks/statystyka/inne/eeg/eeg_files/20241017_kp_cleaned.edf"
raw = mne.io.read_raw_edf(raw_path,
                          preload=True,exclude=["EXG1","EXG2","EXG3","EXG4","EXG5"])

raw.set_montage("biosemi64")

# Wybieramy tylko kanały EEG
picks = mne.pick_types(raw.info, eeg=True, exclude=[])
dane = raw.get_data(picks=picks)

# Aktualizujemy info, aby zawierało tylko wybrane kanały
info = mne.pick_info(raw.info, picks)

"""Uproszczony model głowy"""
sphere_model = mne.make_sphere_model(info=info)

"""Tworzenie siatki dipoli"""
src = mne.setup_volume_source_space(subject=None, pos=3)

"""Rozwiązanie forward problem"""
fwd = mne.make_forward_solution(info,
                               eeg=True,
                               meg=False,
                               trans=None,
                               src=src,
                               bem=sphere_model)

"""W celu uproszczenia"""
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)
macierz_zyskow = fwd_fixed['sol']['data']

print("Wymiary macierzy zysków:", macierz_zyskow.shape)
dipoles = macierz_zyskow.shape[1]
alpha = 0.1
dane_punkt = dane[:, 200:201]



def montecarlo(n_samples,n_iterations):
    Q = np.zeros((dipoles, n_iterations))
    for i in range(n_iterations):
        wybrane_dipole = np.random.choice(dipoles, size=n_samples, replace=False)
        macierz_zyskow_przeskal = macierz_zyskow[:, wybrane_dipole] * np.sqrt(dipoles/n_samples)
        alpha = 0.1
        R_odwr = np.eye(macierz_zyskow_przeskal.shape[1]) * alpha
        G_wMNE = macierz_zyskow_przeskal.T @ macierz_zyskow_przeskal + R_odwr
        Q_red = np.linalg.inv(G_wMNE) @ macierz_zyskow_przeskal.T @ dane_punkt
        Q_full = np.zeros(dipoles)
        Q_full[wybrane_dipole] = Q_red.flatten()
        Q[:,i] = Q_full
    return np.mean(Q, axis=1)



n_samples = dipoles
n_iterations = 1000
start_mc = time.time()
Q_monte_carlo = montecarlo(n_samples=n_samples, n_iterations=n_iterations)
time_mc = time.time() - start_mc
print(f"Czas rozwiązania metodą Monte Carlo: {time_mc:.4f} s")
