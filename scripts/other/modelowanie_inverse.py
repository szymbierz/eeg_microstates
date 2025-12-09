import mne
from autoreject import AutoReject
import numpy as np 
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
"""ZAIMPORTOWANIE PLIKÓW"""
import autoreject
ica_fname = "/Users/szymbierz/Desktop/wszystko/notebooks/statystyka/inne/eeg/best_ica_19.05-ica.fif"
epochs_ar_fname = "/Users/szymbierz/Desktop/wszystko/notebooks/statystyka/inne/eeg/epochs_ar_1.set"
reject_log_filename = "/Users/szymbierz/Desktop/wszystko/notebooks/statystyka/inne/eeg.npz"
ica = mne.preprocessing.read_ica(ica_fname)
epochs_full = mne.read_epochs_eeglab(epochs_ar_fname)
reject_log = autoreject.read_reject_log(reject_log_filename)
epochs_ica = epochs[~reject_log.bad_epochs]
ica.apply(epochs_full,exclude=ica.exclude)
epochs_main = epochs_full.copy()
