import mne
import numpy as np
from class_microstates import Microstates
def segment_epochs(epochs,model:Microstates):
    """Funkcja, która iteracyjne wywołuje metodę predict na danych w obrębie i-tej epoki.
    Uzyskuje się sekwencję etykiet mikrostanów dla każdej próbki w obrębie i-tej epoki.
    Funkcja zwraca końcową sekwencję etykiet mikrostanów dla wszystkich próbek w obrębie wszystkich epok.
    Funkcja zwraca dane epok wraz z odpowiadającymi im etykietami mikrostanów.
    """

    if epochs.get_data().ndim != 3:
        raise ValueError("Błąd: data nie ma wymiaru 3")
    if model.centroids is None:
        raise RuntimeError("Błąd: model nie został wytrenowany")
    
    n_epochs,n_channels,n_samples = epochs.get_data().shape #pobranie wymiarów danych epok
    epochs_data = epochs.get_data() #pobranie danych epok

    if model.algorithm == "kmeans++":

        labels_per_epoch = [] #lista przechowująca etykiety mikrostanów dla epok
        data_list = [] #lista przechowująca dane epok

        for e in range(n_epochs):
            current_epoch = epochs_data[e,:,:] #pobranie danych epoki
            current_labels = model.predict(current_epoch) #uzyskanie etykiet mikrostanów dla epoki
            labels_per_epoch.append(current_labels) #dodanie etykiet do listy
            data_list.append(current_epoch) #dodanie danych epoki do listy -> spójność danych z etykietami 

        final_labels = np.concatenate(labels_per_epoch) #połączenie etykiet epok w jeden wektor
        final_data = np.hstack(data_list) #połączenie danych z poszczególnych epok w jedną macierz 
    
        return final_labels,final_data
    
    elif model.algorithm == "fcm" or model.algorithm == "fkmeans":

        U_per_epoch = [] #lista przechowująca macierze przynależności dla epok
        data_list = [] #lista przechowująca dane epok

        for e in range(n_epochs):
            current_epoch = epochs_data[e,:,:] #pobranie danych epoki
            current_U = model.predict(current_epoch) #uzyskanie macierzy przynależności dla epoki
            U_per_epoch.append(current_U) #dodanie macierzy przynależności do listy
            data_list.append(current_epoch) #dodanie danych epoki do listy -> spójność danych z macierzami przynależności

        final_U = np.hstack(U_per_epoch) #połączenie macierzy przynależności z poszczególnych epok w jedną macierz 
        final_data = np.hstack(data_list) #połączenie danych z poszczególnych epok w jedną macierz 
        hard_labels = np.argmax(final_U,axis=0) #uzyskanie etykiet mikrostanów dla każdej próbki

        return hard_labels,final_data,final_U

  





    



