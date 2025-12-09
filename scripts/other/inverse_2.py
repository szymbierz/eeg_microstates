import mne 
import numpy as np
import matplotlib.pyplot as plt 

# Ustawienia podstawowe - używamy prawdziwych danych EEG
raw_path = "/Users/szymbierz/Desktop/wszystko/notebooks/statystyka/inne/eeg/eeg_files/20241017_kp_cleaned.edf"
raw = mne.io.read_raw_edf(raw_path, preload=True, exclude=["EXG1","EXG2","EXG3","EXG4","EXG5"])
raw.set_montage("biosemi64")
raw.crop(0, 60)  # weź tylko 1 minutę danych
raw.filter(8, 40)  # bandpass dla lepszych wyników

# Model sfery głowy
sphere_model = mne.make_sphere_model(info=raw.info)


src = mne.setup_volume_source_space(subject=None, pos=8.0)  # 8mm spacing
print(f"Liczba dipoli: {src[0]['nuse']}")

# Forward solution
fwd = mne.make_forward_solution(raw.info, trans=None, src=src, 
                               bem=sphere_model, eeg=True, meg=False)
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)

# Macierz zysków (Lead field matrix)
G = fwd_fixed['sol']['data']  # shape: (n_channels, n_dipoles)
print(f"Wymiary macierzy zysków: {G.shape}")

# Obliczanie macierzy kowariancji danych
cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)

# Implementacja prostej metody Monte Carlo do redukcji liczby dipoli
# Założenie: wybieramy losowy podzbiór dipoli do analizy
n_dipoles_total = G.shape[1]
n_dipoles_mc = 100  # Przykładowa, mała liczba dipoli dla uproszczenia
if n_dipoles_total < n_dipoles_mc:
    n_dipoles_mc = n_dipoles_total # Jeśli wszystkich dipoli jest mniej niż zakładana próbka MC

# Losowy wybór indeksów dipoli
np.random.seed(42) # dla powtarzalności wyników
selected_indices = np.random.choice(n_dipoles_total, n_dipoles_mc, replace=False)

# Redukcja macierzy zysków do wybranych dipoli
G_mc = G[:, selected_indices]
print(f"Wymiary macierzy zysków po redukcji Monte Carlo: {G_mc.shape}")

# Przygotowanie do rozwiązania problemu odwrotnego (np. MNE/dSPM)
# Dla uproszczenia, użyjemy prostego estymatora MNE (Minimum Norm Estimate)
# Tworzymy "sztuczny" obiekt inverse_operator, aby móc użyć funkcji MNE
# W praktyce, potrzebowalibyśmy pełnej definicji inverse_operatora
# Tutaj symulujemy go, aby móc przejść do wizualizacji na podstawie G_mc

# Dla demonstracji stworzymy sztuczne dane evoced
# Wybieramy tylko kanały EEG do stworzenia obiektu Evoked
picks_eeg = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, exclude=raw.info['bads'])
eeg_data_for_evoked = raw.get_data(picks=picks_eeg)
evoked_data_mean = np.mean(eeg_data_for_evoked, axis=1) # uśrednione dane EEG
info_for_evoked = mne.pick_info(raw.info, picks_eeg) # Info tylko dla kanałów EEG

# Sprawdzenie wymiarów przed utworzeniem EvokedArray
# print(f"Shape of eeg_data_for_evoked: {eeg_data_for_evoked.shape}")
# print(f"Shape of evoked_data_mean: {evoked_data_mean.shape}")
# print(f"Number of channels in info_for_evoked: {info_for_evoked['nchan']}")

times = np.linspace(0, raw.times[-1], evoked_data_mean.shape[0] if evoked_data_mean.ndim == 1 else evoked_data_mean.shape[1]) # Poprawka dla times
evoked = mne.EvokedArray(evoked_data_mean[:, np.newaxis], info_for_evoked, tmin=0)

# Rozwiązanie problemu odwrotnego dla zredukowanej liczby dipoli (symulacja)
# W typowym scenariuszu użylibyśmy mne.minimum_norm.apply_inverse_epochs lub podobnej
# Tutaj, dla prostoty, "oszacujemy" aktywność źródłową proporcjonalnie do rzutowania danych na macierz G_mc
# To jest DUŻE uproszczenie i nie jest to standardowa metoda!
# Celem jest pokazanie wizualizacji, a nie dokładności rozwiązania odwrotnego.

# "Estymacja" aktywności źródłowej (bardzo uproszczona)
# Używamy pseudo-inwersji Moore'a-Penrose'a
lambda2 = 0.1 # Parametr regularyzacji
G_mc_inv = np.linalg.pinv(G_mc.T @ G_mc + lambda2 * np.eye(G_mc.shape[1])) @ G_mc.T
estimated_activity_mc = G_mc_inv @ evoked.data[:,0] # Aktywność dla pierwszego punktu czasowego

print(f"Oszacowana aktywność dla {n_dipoles_mc} dipoli: {estimated_activity_mc.shape}")

# Przygotowanie do wizualizacji 3D
# Potrzebujemy pozycji wybranych dipoli
src_mc = src[0].copy() # Kopiujemy strukturę source space
src_mc['rr'] = src[0]['rr'][selected_indices]
src_mc['nn'] = src[0]['nn'][selected_indices]
src_mc['nuse'] = len(selected_indices)
src_mc['vertno'] = np.arange(len(selected_indices)) # Indeksy dla nowych wierzchołków
src_mc['inuse'] = np.ones(len(selected_indices), dtype=int) # Wszystkie wybrane są w użyciu
src_mc['id'] = src[0]['id'] # Zachowujemy oryginalne ID, jeśli potrzebne
# Upewniamy się, że 'nearest' i 'dist' są odpowiednio zaktualizowane, jeśli istnieją i są używane
# Dla wolumetrycznego source space, te pola mogą nie być krytyczne jak dla powierzchniowego
if 'nearest' in src_mc and src_mc['nearest'] is not None:
    # To jest bardziej skomplikowane dla wolumetrycznego i może nie być potrzebne
    # dla prostej wizualizacji pozycji i aktywności
    pass
if 'dist' in src_mc and src_mc['dist'] is not None:
    pass


# Tworzenie obiektu SourceEstimate dla wizualizacji
# Potrzebujemy "czasów" dla naszego pojedynczego oszacowania
stc_times = np.array([0.0]) # jeden punkt czasowy
# Tworzymy dane dla SourceEstimate: (n_vertices, n_times)
stc_data = estimated_activity_mc[:, np.newaxis]

# Ważne: MNE oczekuje, że 'vertices' w SourceEstimate to lista arrayów,
# po jednym dla każdego hemisfery (lh, rh) dla źródeł powierzchniowych.
# Dla źródeł wolumetrycznych (jak nasze setup_volume_source_space),
# oczekuje pojedynczego arraya z indeksami wierzchołków użytych w src.
# W naszym przypadku, po selekcji Monte Carlo, 'vertno' w src_mc już zawiera te indeksy.
stc = mne.VolSourceEstimate(stc_data, vertices=[src_mc['vertno']], tmin=0, tstep=1, subject=None)


# Wizualizacja 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Pozycje wszystkich dipoli (dla kontekstu)
ax.scatter(src[0]['rr'][:,0], src[0]['rr'][:,1], src[0]['rr'][:,2], c='gray', alpha=0.1, label='Wszystkie dipole')

# Pozycje i aktywność wybranych dipoli
# Normalizujemy aktywność dla lepszej wizualizacji kolorów
activity_norm = (estimated_activity_mc - np.min(estimated_activity_mc)) / (np.max(estimated_activity_mc) - np.min(estimated_activity_mc) + 1e-6)
scatter = ax.scatter(src_mc['rr'][:,0], src_mc['rr'][:,1], src_mc['rr'][:,2], c=activity_norm, cmap='hot', s=50, label='Aktywne dipole (MC)')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Zrekonstruowana aktywność dipoli (Metoda Monte Carlo - uproszczona)')
fig.colorbar(scatter, label='Znormalizowana aktywność')
plt.legend()
plt.show()



