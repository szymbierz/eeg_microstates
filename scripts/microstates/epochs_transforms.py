import mne
import numpy as np

def reduce_dim(epochs:object):
    """Funkcja, która zmienia wymiar macierzy danych epok eeg
    Funkcja przyjmuje obiekt klasy Epochs o wymiarach (n_epochs,n_channels,n_samples)
    Funkcja zwraca zmodyfikowany obiekt klasy epochs o wymiarach (n_channels,n_samples)
    """
    if not isinstance(epochs,mne.Epochs):
        raise ValueError("Obiekt nie jest obiektem klasy Epochs") #sprawdzenie czy wczytany obiekt jest obiektem klasy Epochs
    if epochs.get_data().ndim != 3:
        raise ValueError("Obiekt nie ma wymiaru 3") #sprawdzenie czy obiekt ma wymiar 3
    
    data = epochs.get_data() #pobranie danych z obiektu
    n_epochs,n_channels,n_samples = data.shape 
    return data.transpose(1,0,2).reshape(n_channels,n_samples*n_epochs)


#def handle_labels_epochs_borders(labels:np.ndarray,epochs_reduced:np.darray):
    """Funkcja, która zamienia etykiety na granicach epok na -1
    Utworzona, aby nie uwzględniać mikrostanów na granicach epok
    Zapobiega to nieprawidłowym
    """




