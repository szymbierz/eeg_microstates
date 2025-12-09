import numpy as np 
import mne 
from itertools import groupby
class MicrostateMetrics:
    
    """
    Klasa do obliczania metryk mikrostanów.
    Olbliczane metryki:
    - Średni czas trwania mikrostanu ✔
    - Częstotliwość występowania mikrostanu ✔
    - Procentowy zakres czasu trwania mikrostanu 
    - Global Explained Variance (GEV)
    - Prawdopodobieństwo przejść 
    """

    def __init__(
            self,
            labels:np.ndarray,
            data:np.ndarray,
            info:mne.Info,
            centroids:np.ndarray,
            U:np.ndarray = None 
            ) -> None: 

        self.labels = labels #etykiety mikrostanów
        self.data = data #dane eeg -> epoki po segmentacji
        self.sfreq = info["sfreq"] #częstotliwość próbkowania
        self.centroids = centroids #dane centroidów mikrostanów (wzorcowe)
        self.microstate_labels = np.unique(labels) #zmienna przechowująca informację o tym jakie mikrostany występują w danych 
        self.n_microstates = len(self.microstate_labels) #liczba zarejestrowanych mikrostanów
        self.labels = self.labels.astype(int) #zamiana na int, aby uniknąć błędu związanego z typem danych
        self.U = U 
        self.gfp = np.std(data,axis=0) #global field power
        self.labels_segments = [(label,sum(1 for _ in group)) for label,group in groupby(self.labels)] #lista przechowująca krotki dotyczące długości występujących po sobie segementów etykiet (mikrostanów) -> (etykieta,długość) 
        #Obsługa błędnych danych
        if self.data.ndim !=2:
            raise ValueError(
                "Błąd: dane wejściowe mają nieodpowiedni wymiar \n"
                "Wymagany wymiar: 2, \n"
                f"Wymiar danych wejściowych: {self.data.ndim} "
                )
        
        if self.centroids is None or self.centroids.ndim !=2:
            raise ValueError(
                "Błąd: dane centroidów są nieprawidłowe lub model nie został wytrenowanyt\n"
                "Wymagany wymiar: 2, \n"
                f"Wymiar danych centroidów: {self.centroids.ndim} "
                )
        
        if self.labels.ndim !=1:
            raise ValueError(
                "Błąd: dane etykiet mają nieodpowiedni wymiar \n"
                "Wymagany wymiar: 1, \n"
                f"Wymiar danych etykiet: {self.labels.ndim} "
                )
        
        if self.data.shape[1] != len(self.labels):
            raise ValueError(
                "Błąd: niezgodność liczby danych z liczbą etykiet \n"
                f"Liczba danych eeg: {self.data.shape[1]}, \n"
                f"Liczba etykiet: {len(self.labels)} "
                )
        
        if self.data.shape[0] != self.centroids.shape[0]:
            raise ValueError(
                "Błąd: niezgodność liczby kanałów danych z liczbą kanałów centroidów \n"
                f"Liczba kanałów danych: {self.data.shape[0]}, \n"
                f"Liczba kanałów centroidów: {self.centroids.shape[0]} "
                )


    def calculate_duration(self):
        """
        Funkcja obliczająca średni czas trwania mikrostanu
        Funkcja zwraca słownik zawierający średni czas trwania mikrostanu dla każdego mikrostanu
        """

        """Do labels-segment wykorzystano funkcję groupby z modułu itertools
        groupby wykrywa sekwencje powtarzających się elementów w liście
        group jest iterator, stąd sprawdzenie długości sekwencji -> sum(1 for _ in group)
        """
        microstates_duration = {} #słownik przechowujący średni czas trwania mikrostanu dla każdego mikrostanu

        for m_label in self.microstate_labels:
            current_microstate_durations = [] #lista przechowująca długość mikrostanu dla m-label stanu 
            for label,duration in self.labels_segments: #iteracja po wszystkich segmentach etykiet 
                if label == m_label:
                    current_microstate_durations.append(duration) #dodanie długości segmentu do listy 
            if current_microstate_durations:
                mean_microstates_duration = np.mean(current_microstate_durations) #obliczenie średniej 
                microstates_duration[m_label] = mean_microstates_duration/self.sfreq #zapisanie średniej długości m_label mikrostanu do słownika 

        return microstates_duration
    

    def coverage(self):
        """Funkcja obliczająca pokrycie czasowe mikrostanu
        Funkcja zwraca słownik zawierający pokrycie czasowe dla danego mikrostanu
        """

        unique_labels,counts = np.unique(self.labels,return_counts=True) #zwraca -> ilość wystąpień każdego mikrostanu w danych 

        coverage = (counts/len(self.labels))*100 #obliczenie pokrycia czasowego dla każdego mikrostanu

        return dict(zip(unique_labels,coverage))
    
    def GEV(self):
        """Funkcja obliczająca Global Explained Variance (GEV)
        Funkcja zwraca wartość GEV
        """

        numerator = 0 #licznik
        denominator = 0 #mianownik
        
        for t_point in range(self.data.shape[1]): #w zakresie puntków czasowych
            current_centroid = self.centroids[:,self.labels[t_point]] #wybór dla danego punktu czasowego 
            current_data = self.data[:,t_point] #wybór dla danego punktu czasowego
            correlation = np.corrcoef(current_data,current_centroid)[0,1] #obliczenie korelacji pomiędzy sygnałem a centroidem
            current_gfp = self.gfp[t_point] #obliczenie globalnej mocy sygnału dla danego punktu czasowego

            numerator += (correlation * current_gfp) ** 2
            denominator += current_gfp ** 2 

        gev = numerator / denominator #licznik przez mianownik -> GEV 

        return gev * 100


    def microstates_per_second(self):
        """Funkcja obliczająca średnie wystąpienie danego mikrostanu na sekundę
        Funkcja zwraca słownik zawierający średnie wystąpienie danego mikrostanu na sekundę dla każdego mikrostanu
        """

        occurences_dict = {} #słownik przechowujący liczbę wystąpień danego mikrostanu
        total_time = len(self.labels)/self.sfreq #całkowity czas 

        for m_label in self.microstate_labels: #iteracja po wszystkich etykietach mikrostanów
            n_m_label_segment = sum(1 for label,_ in self.labels_segments if label == m_label) #zliczenie ilości segmentów mikrostanu m_label
            occurences_dict[m_label] = n_m_label_segment / total_time 

        return occurences_dict



    def transtition_propability(self):
        """Funkcja obliczająca prawdopodobieństwo przejścia pomiędzy mikrostanami
        Funkcja zwraca macierz prawdopodobieństw przejścia
        """

        transition_matrix = np.zeros((self.n_microstates,self.n_microstates)) #macierz przejść pomiędzy mikrostanami, bez przejść pomiędzy stanami o tej samej etykiecie

        sequences_of_states = [label for label,_ in self.labels_segments] #lista sekwencji etykiet mikrostanów 

        for i in range(len(sequences_of_states) - 1 ):
            state_from = sequences_of_states[i]
            state_to = sequences_of_states[i+1]
            transition_matrix[state_from,state_to] += 1

        row_sums = transition_matrix.sum(axis=1,keepdims=True) #suma zliczen przejsc w kazdym wierszu
        row_sums_non_zero = np.where(row_sums==0,1,row_sums) #zastąpienie zera przez 1, aby uniknąć dzielenia przez zero (jeśli gdzieś nie było przejść)
        propability_transtiton_matrix = transition_matrix / row_sums_non_zero 
        return propability_transtiton_matrix
    

    def fuzzy_metrics(self):
        """Funkcja obliczająca metryki specyficzne dla mikrostanów otrzymanych z algorytmu Fuzzy C-Means
        Funkcja zwraca słownik metryk:
        - Średnia przynależność do mikrostanu
        - Entropia przynależności do mikrostanu
        - Niepewność klasyfikacji 
        """
        if self.U is None:
            raise ValueError("Błąd: macierz przynależności nie została przekazana")
        
        fuzzy_metrics = {}

        # Średnia przynależność dla każdego mikrostanu
        mean_membership = np.mean(self.U, axis=1)
        fuzzy_metrics['mean_membership'] = dict(zip(range(self.n_microstates), mean_membership))

        # Entropia przynależności do mikrostanu
        entropy = -np.sum(self.U * np.log(self.U + 1e-10), axis=0)
        fuzzy_metrics['entropy'] = np.mean(entropy)

        # Niepewność klasyfikacji
        uncertainty = 1 - np.max(self.U, axis=0)
        fuzzy_metrics['uncertainty'] = np.mean(uncertainty)

        return fuzzy_metrics




