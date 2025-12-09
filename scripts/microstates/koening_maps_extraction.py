

import numpy as np 
import mne 
from visualise_microstates import visualise_base_microstates

ch_koenig_path = "/Users/szymbierz/Desktop/wszystko/notebooks/statystyka/inne/eeg/Dane/processed/mean_models_koenig_et_al_2002_chlist.asc"
centroids_koenig_path = "/Users/szymbierz/Desktop/wszystko/notebooks/statystyka/inne/eeg/Dane/processed/mean_models_koenig_et_al_2002.asc"

koenig_orig_ch = np.loadtxt(ch_koenig_path,dtype=str)
koenig_centroids = np.loadtxt(centroids_koenig_path,dtype=float)

channel_types = ["eeg"] * len(koenig_orig_ch)
sfreq = 1 

# print(
#     f"Nazwy kanałów: {koenig_orig_ch}\n",
#     f"Maacierz centroidów:{koenig_centroids}\n",
#     f"Kształt macierzy centroidów: {koenig_centroids.shape}\n",
# )

montage = mne.channels.make_standard_montage('standard_1020')

# info = mne.create_info(
#     ch_names = koenig_orig_ch.tolist(),
#     ch_types = channel_types,
#     sfreq = sfreq,
#     )
# info.set_montage(montage, match_case=False, on_missing='ignore')


# visualise_base_microstates(
#     base_microstates=koenig_centroids,
#     figsize=(15, 4),
#     info=info,
#     show=True,
# )


my_ch_names = [
'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1',
'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
]

my_ch_set = set(my_ch_names)
koenig_ch_list = koenig_orig_ch.tolist()
koenig_ch_set = set(koenig_ch_list)


common_channels = [ch for ch in koenig_ch_set if ch in my_ch_set]

my_ch_index_map = {name:index for index,name in enumerate(my_ch_names)}
my_indices = [my_ch_index_map[ch] for ch in common_channels]

koenig_ch_index_map = {name:index for index,name in enumerate(koenig_ch_list)}
koenig_indices = [koenig_ch_index_map[ch] for ch in common_channels]

# print(my_indices)
# print(koenig_indices)

for_corr_pyprep = np.load("/Users/szymbierz/Desktop/wszystko/notebooks/statystyka/inne/eeg/scripts/microstates/maps_pyprep.npy")
for_corr_autoreject = np.load("/Users/szymbierz/Desktop/wszystko/notebooks/statystyka/inne/eeg/scripts/microstates/maps_autoreject.npy")


my_maps_pyprep_corr = for_corr_pyprep[:,my_indices]
my_maps_autoreject_corr = for_corr_autoreject[:,my_indices]

koenig_maps_corr = koenig_centroids[:,koenig_indices]

# print(my_maps_pyprep_corr.shape)
# print(my_maps_autoreject_corr.shape)
# print(koenig_maps_corr.shape)

# info_2 = mne.create_info(
#     ch_names = common_channels,
#     ch_types = channel_types,
#     sfreq = sfreq,
#     )
# info_2.set_montage(montage, match_case=False, on_missing='ignore')

# visualise_base_microstates(
#     base_microstates=my_maps_autoreject_corr,
#     figsize=(15, 4),
#     info=info_2,
#     show=True,
# )
correlation_matrix = np.zeros((4,4))

for i in range(4):
    for j in range(4):

        my_map = my_maps_autoreject_corr[i,:]
        koenig_map = koenig_maps_corr[j,:]

        correlation = np.abs(np.corrcoef(my_map,koenig_map)[0,1])
        correlation_matrix[i,j] = correlation



import seaborn as sns
import matplotlib.pyplot as plt

my_map_labels = ['Microstate 1', 'Microstate 2', 'Microstate 3', 'Microstate 4']

# (Zakładając, że kanoniczne są w kolejności A,B,C,D)
canonical_map_labels = ['A', 'B', 'C', 'D']

# Narysuj heatmapę
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix, 
    annot=True,          # Pokaż wartości liczbowe na polach
    fmt=".2f",           # Formatuj liczby do 2 miejsc po przecinku
    cmap='viridis',      # Wybierz ładną mapę kolorów
    xticklabels=canonical_map_labels,
    yticklabels=my_map_labels,
    vmin=0,              # Ustaw minimum skali kolorów na 0
    vmax=1               # Ustaw maksimum skali kolorów na 1
)
plt.title('Spatial Correlation with Canonical Maps')
plt.xlabel('Canonical Microstates')
plt.ylabel('My Preprocessed Microstates')
plt.show()

print(correlation_matrix)