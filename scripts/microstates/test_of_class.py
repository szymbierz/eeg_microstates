import numpy as np 
import mne

from scripts.microstates.class_microstates import Microstates

from class_MicrostateMetrics import MicrostateMetrics 
from scipy.signal import find_peaks 
import matplotlib.pyplot as plt 
import autoreject

from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs

"""ZAŁADOWANIE SYGNAŁU I USTAWIENIE UKŁADU ELEKTROD"""
d_path = "eeg_files/20241017_kp.bdf" #ścieżka do pliku
ch_eog = ["EXG1","EXG2"] #zdefiniowanie dedykowanych kanałów EOG
ch_ecg = ["EXG3","EXG4"] #zdefiniowanie dedykowanych kanałów ECG
ch_exclude = [f"EXG{i}" for i in range(5,9)] #wykluczenie kanałów niewykorzystanych w badaniu
raw = mne.io.read_raw_bdf(
    d_path, preload = True, eog = ch_eog, misc = ch_ecg, exclude = ch_exclude 
)
raw.set_montage("biosemi64")
raw.set_eeg_reference(projection=True)
raw.filter(0.1, 45)
raw.notch_filter(50)
epochs = mne.make_fixed_length_epochs(raw, duration=4, preload=True)
epochs.load_data()


ica_fname = "/Users/szymbierz/Desktop/wszystko/notebooks/statystyka/inne/eeg/best_ica_19.05-ica.fif"
epochs_ar_fname = "/Users/szymbierz/Desktop/wszystko/notebooks/statystyka/inne/eeg/epochs_ar_1.set"
reject_log_filename = "/Users/szymbierz/Desktop/wszystko/notebooks/statystyka/inne/eeg.npz"
ica = mne.preprocessing.read_ica(ica_fname)
epochs_ar = mne.read_epochs_eeglab(epochs_ar_fname)
reject_log = autoreject.read_reject_log(reject_log_filename)
epochs_ica = epochs[~reject_log.bad_epochs]


ica.exclude=[0,1,54] 
epochs_ica_cleaned_1 = epochs_ar.copy()


ica.apply(epochs_ica_cleaned_1, exclude=ica.exclude)
# Lista typowych kanałów EEG (np. z dokumentacji Biosemi64)
biosemi64_names = mne.channels.make_standard_montage('biosemi64').ch_names


epochs_eeg_final_1 = epochs_ica_cleaned_1.copy().pick_channels(biosemi64_names)
epochs_eeg_final_1.set_eeg_reference("average")

print(epochs_eeg_final_1.ch_names)
print(len(epochs_eeg_final_1.ch_names))

all_peaks = []

# Przechodzimy po epokach w epochs_eeg_final_1
for epoch in epochs_eeg_final_1.get_data():  # get_data() -> shape: (n_epochs, n_channels, n_times)
    gfp = np.std(epoch, axis=0)  # GFP dla tej epoki
    peak_idxs, _ = find_peaks(gfp)
    # Wyciągamy mapy topograficzne dla tych pików
    maps = epoch[:, peak_idxs]  # shape: (n_channels, n_peaks_in_epoch)
    all_peaks.append(maps)




peaks_all = np.hstack(all_peaks)  # shape: (n_channels, n_peaks_total)


import time 

micro = Microstates(peaks=peaks_all, n_microstates=4, max_iters=100)

print("Start inicjalizacji ")

start = time.time()
micro.fit(peaks_all)
end = time.time()

print(f"Czas inicjalizacji i fitowania mikrostanów: {end - start:.2f} sekund")