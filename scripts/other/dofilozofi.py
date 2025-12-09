import mne
import numpy as np
import matplotlib.pyplot as plt

# --- KROK 1. Wczytanie danych ---
ch_eog = ["EXG1", "EXG2"]   # ruchy gałek ocznych
ch_ecg = ["EXG3", "EXG4"]   # kanały EKG (jeśli występują)
ch_exclude = [f"EXG{i}" for i in range(5, 9)]
file = "eeg_files/20241017_kp_cleaned.edf"

# Wczytanie surowych danych
raw = mne.io.read_raw_edf(
    file,
    preload=True,
    eog=ch_eog,
    misc=ch_ecg,
    exclude=ch_exclude
)

# Ustawienie montażu (rozmieszczenia elektrod)
montage = "biosemi64"
raw.set_montage(montage=montage)

raw.crop(tmax=10)
raw.plot()
plt.show()
