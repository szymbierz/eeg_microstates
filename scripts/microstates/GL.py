# Funkcja do generowania obrazka GFP (Global Field Power) dla danych EEG w Pythonie
# Zakładamy, że eeg_data to numpy array o wymiarach (channels, timepoints)
import numpy as np
import matplotlib.pyplot as plt

def plot_gfp(eeg_data, sampling_rate, ax=None):
    """
    Rysuje Global Field Power (GFP) dla danych EEG.

    Parameters
    ----------
    eeg_data : np.ndarray
        Dane EEG o wymiarach (channels, timepoints)
    sampling_rate : float
        Częstotliwość próbkowania w Hz
    ax : matplotlib.axes.Axes, optional
        Osie, na których narysować wykres. Jeśli None, tworzy nowe.
    """
    gfp = np.std(eeg_data, axis=0)  # STD przez kanały, dla każdego punktu w czasie
    times = np.arange(gfp.size) / sampling_rate

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, gfp, color='blue', lw=2)
    ax.set_xlabel("Czas (s)")
    ax.set_ylabel("GFP (μV)")
    ax.set_title("Lokalna moc pola (GFP) w EEG")
    ax.grid(True)
    plt.tight_layout()
    if ax is None:
        plt.show()

if __name__ == "__main__":
    eeg_data = np.random.randn(32, 1000)  # 32 kanały, 1000 punktów w czasie
    plot_gfp(eeg_data, sampling_rate=250)
    plt.savefig("gfp.png")  # zapis wykresu do pliku w tym samym katalogu
    plt.show()


