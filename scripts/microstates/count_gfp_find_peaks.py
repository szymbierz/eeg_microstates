import numpy as np
from scipy.signal import find_peaks

def count_GFP(eeg_data,distance):
    """Funkcja oblicza GFP oraz znajduje lokalne maksima GFP
    argumenty: 
    eeg_data - dane eeg (n_kanałów,n_próbek), 
    distance - odległość pomiędzy lokalnymi maksimami GFP
    zwraca: 
    peaks - dane lokalnych map szczytów GFP
    """
    if eeg_data.shape[0] > eeg_data.shape[1]: #obsługa błędu, gdy dane: (n_próbek,n_kanałów)
        raise ValueError(f"Niepoprawny kształy danych: {eeg_data.shape}\n "
                         f"Wymagana transpozycja danych (data.T)")
    GFP = np.std(eeg_data,axis=0) #axis=0 - po kolumnach
    gfp_peaks,_ = find_peaks(GFP,distance=distance) #szukanie maksim GFP
    peaks = eeg_data[:,gfp_peaks] #przypisanie indeksów gfp_peaks do wejściowych danych
    print("Poprawnie uzyskano mapy topograficzne sygnału eeg.")
    return peaks 
    

"""Do poprawy"""