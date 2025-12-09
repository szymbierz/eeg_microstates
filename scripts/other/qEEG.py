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

# --- KROK 2. Definicja pasm częstotliwości ---
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta':  (12, 35),
    'Gamma': (30, 100)
}

# Wybierz kanał, który chcesz zilustrować:
channel_to_plot = 'O1'  # np. kanał potyliczny; zmień na dowolny inny

# --- KROK 3. Rysowanie ---
fig, axs = plt.subplots(len(bands), 1, figsize=(6, 8), sharex=True)


for ax, (band_name, (fmin, fmax)) in zip(axs, bands.items()):
    # Kopiujemy surowe dane i filtrujemy je w zadanym paśmie
    raw_band = raw.copy().filter(fmin, fmax, fir_design='firwin', verbose=False)
    
    # Pobieramy dane z wybranego kanału
    data, times = raw_band[channel_to_plot]
    # data.shape = (1, n_samples), times.shape = (n_samples,)
    data = data[0]  # konwersja do wektora (n_samples,)

    # Dla przejrzystości wyświetlamy tylko 2 sekundy (możesz zmienić na inny wycinek)
    sfreq = raw_band.info['sfreq']
    t_max = 2.0  # czas w sekundach
    n_samples = int(t_max * sfreq)

    data_snippet = data[:n_samples]
    time_snippet = times[:n_samples]

    ax.plot(time_snippet, data_snippet, label=f"{band_name} ({fmin}-{fmax} Hz)")
    ax.set_ylabel("Amplituda [µV]")
    ax.grid(True)
    ax.legend(loc="upper right")

axs[-1].set_xlabel("Czas [s]")
plt.tight_layout()
plt.show()