import numpy as np 



class Microstates:
    """Klasa do analizy mikrostanów EEG.
    Klasa wykorzystuje klasteryzację, 
    która bazuje na podstawie implementacji algorytmu k-means++.
    Zaimplementowany algorytm k-means++ pozwala na grupowanie map topograficznych
    sygnału EEG w mikrostany.
    Algorytm dostosowany jest w taki sposób, 
    aby nie uwzględniał polaryzacji map topograficznych
    """
    def __init__(
            self,
            peaks: np.ndarray, #dane eeg z uzyskanymi mapami topograficznymi (n_channels,n_peaks)
            n_microstates: int, #liczba mikrostanów do wyodrębnienia proporcjonalna liczbie klastrów (k)
            max_iters: int,
            tol: float= 1e-4, #tolerancja zbieżnosci 
            algorithm: str = "kmeans++", #wybór algorytmu 
            m: float = 2 #parametr rozymcia (dla fuzzy c means)
        ) -> None:
        self.peaks = peaks
        self.n_microstates = n_microstates
        self.max_iters = max_iters
        self.tol = tol
        self.algorithm = algorithm
        self.m = m 
        self.centroids = None
        self.labels = None 
        self.U = None #macierz przynależności do mikrostanów (dla fuzzy c means) 
        
    
    

    def centroids_initialisation(self,peaks): #funkcja inicjalizująca centroidy
        if not isinstance(peaks,np.ndarray): #obsługa błedu związanego z typem danych
            raise TypeError("Błąd: dane nie są tablicą numpy")
        
        if peaks.ndim == 1 or peaks.ndim > 3: #obsługa błedu związanego z wymiarami macierzy danych eeg
            raise ValueError(f"Błąd: dane mają nieprawidłowe wymiary. Wymiar danych to: {peaks.ndim}")
        
        n_channels,n_peaks = peaks.shape #rozpakowanie wymiarów macierzy danych eeg
        if self.n_microstates > n_peaks: #obsługa błędu zgodności liczby mikrostanow z liczbą map topograficznych
            raise ValueError(
                f"Wybrano nieodpowiednią liczbę mikrostanów : {self.n_microstates} !\n"
                f"Liczba mikrostanów musi być większa od liczby map topograficznych w danch: {n_peaks}\n"
                f"Nalezy dostosować liczbę mikrostanów do liczby map topograficznych w danych"
            )
        
        chosen_centroids = [] #lista przechowująca indeksy wybranych centroidów
        first_centroid_idx = np.random.choice(n_peaks) #wybór losowego indeksu pierwszego centroidu
        chosen_centroids.append(peaks[:,first_centroid_idx].copy()) #dodanie pierwszego centroidu do listy wybranych centroidów
        distances = np.zeros(n_peaks) #tablica przechowująca odległości między centroidami
        
        #wyznaczanie centroidów
        #dla kazdego centroidu, zgodnie z (k-1)
        for i in range(1,self.n_microstates): #centroid 0 został wybrany losowo,stąd zaczynamy od 1
            #dla każdej próbki w n_peaks
            for j in range(n_peaks):
                peak = peaks[:,j]
                min_distance = np.inf #początkowa wartość odległości, która będzie aktualizowana wraz z przebiegiem pętli
                
                #dla każdego centroidu z chosen_centroids
                for c in chosen_centroids:
                    #odległość euklidesowa między próbką a centroidem
                    distance_to_c = np.linalg.norm(peak - c) 
                    #odległość euklidesowa między próbką a odwróconą wartością centroidu (uwzględnienie polaryzacji)
                    distance_to_neg_c = np.linalg.norm(peak - (-c))
                    #kwadrat odległości euklidesowej między próbką a centroidem
                    distance_to_c_sqrt = distance_to_c ** 2 
                    distance_to_neg_c_sqrt = distance_to_neg_c ** 2 
                    #wybór minimalnej odległości euklidesowej między próbką a centroidem
                    min_distance_sqrt = min(distance_to_c_sqrt,distance_to_neg_c_sqrt)

                    if min_distance_sqrt < min_distance:
                        min_distance = min_distance_sqrt 
                distances[j] = min_distance 
            
            #obliczenie prawdopodobieństwa wyboru próbki jako centroidu (KMeans++)
            #jezeli suma odległości wyniosłaby 0 -> aby nie doprowadzic do dzielenia przez 0
            sum_distances = np.sum(distances)
            if sum_distances == 0:
                print("Suma odległości jest równa 0, dokonano losowego wyboru centroidu ")
                #wybór 
                next_centroid_idx = np.random.choice(n_peaks) #bez prawdopodobieństwa
            else:
                #wybór centroidu z prawdpopodbieństwem proporcjonalnym do kwadratu odległości D(x_j)^2
                probabilities = distances / sum_distances 
                next_centroid_idx = np.random.choice(n_peaks,p=probabilities) #wybór centroidu z  obliczonym prawdpopodbieństwem
            #dodanie wybranego centroidu do listy wybranych centroidów
            chosen_centroids.append(peaks[:,next_centroid_idx].copy()) 
        
        #przypisanie wybranych centroidów do atrybutu klasy
        self.centroids = np.array(chosen_centroids) 

    def fit(self,peaks):
        if self.algorithm == "kmeans++":
            self._fit_kmeans(peaks)
        elif self.algorithm == "fcm":
            self._fit_fcm(peaks)
        elif self.algorithm == "fkmeans":
            self._fit_kmeans(peaks)
            self._fit_fcm(peaks,skip_initialisation=True)
    

    def _fit_kmeans(self,peaks):
        if not isinstance(peaks,np.ndarray): #obsługa błedu związanego z typem danych
            raise TypeError("Błąd: dane nie są tablicą numpy")
        
        if peaks.ndim == 1 or peaks.ndim > 3: #obsługa błedu związanego z wymiarami macierzy danych eeg
            raise ValueError(f"Błąd: dane mają nieprawidłowe wymiary. Wymiar danych to: {peaks.ndim}")
        
        n_channels,n_peaks = peaks.shape 


        self.centroids_initialisation(peaks)

        self.labels = np.zeros(n_peaks)

        for i in range(self.max_iters):
            previous_centroids = self.centroids.copy()

            for j in range(n_peaks): #
                peak = peaks[:,j] #peak: (n_kanałów,n_próbek)
                min_distance = np.inf

                for m in range(self.n_microstates): 
                    centroid = self.centroids[m,:] #centroids: (n_mikrostanów,n_kanałów) 
                    distance_to_c = np.linalg.norm(peak - centroid)
                    distance_to_neg_c = np.linalg.norm(peak - (- centroid))
                    distance_to_c_sqrt = distance_to_c ** 2 
                    distance_to_neg_c_sqrt = distance_to_neg_c ** 2
                    min_distance_sqrt = min(distance_to_c_sqrt, distance_to_neg_c_sqrt)
                    
                    #jeśli warunek nie jest spełniony, sprawdzany jest kolejny centroid
                    #jeśli warunek jest spełniony, aktualizacja etykiety i sprawdzenie kolejnego centroidu
                    #po sprawdzeniu wszystkich centroidów, przejście do kolejnej próbki
                    
                    if min_distance_sqrt < min_distance:
                        min_distance = min_distance_sqrt
                        self.labels[j] = m 

            for m in range(self.n_microstates):
                indices = np.where(self.labels == m)[0] #indeksy próbek przypisanych do m-tego mikrostanu
                microstate_peaks = peaks[:,indices] #n_kanałów, n_próbek
                if microstate_peaks.size > 0: #jeśli mikrostan zawiera próbki
                    # Oblicz średnią ze wszystkich punktów w klastrze (z uwzględnieniem polaryzacji)
                    centroid_sum = np.zeros(n_channels)
                    for k in range(microstate_peaks.shape[1]):
                        peak = microstate_peaks[:, k]
                        # Jeżeli iloczyn skalarny jest dodatni, dodaj punkt, jeżeli ujemny, odejmij punkt
                        if np.dot(peak, self.centroids[m,:]) >= 0:
                            centroid_sum += peak
                        else:
                            centroid_sum -= peak
                    
                    new_centroid = centroid_sum / microstate_peaks.shape[1]
                    self.centroids[m,:] = new_centroid / np.linalg.norm(new_centroid)
                else:
                    pass
            diff_of_centroids = np.linalg.norm(self.centroids - previous_centroids) #sprawdzenie zbieżności
            if diff_of_centroids < self.tol: #jeśli zbieżność jest osiągnięta, przerwanie pętli
                break

    def _fit_fcm(self,peaks,skip_initialisation=False):
        if not isinstance(peaks,np.ndarray): #obsługa błedu związanego z typem danych
            raise TypeError("Błąd: dane nie są tablicą numpy")
        
        if peaks.ndim == 1 or peaks.ndim > 3: #obsługa błedu związanego z wymiarami macierzy danych eeg
            raise ValueError(f"Błąd: dane mają nieprawidłowe wymiary. Wymiar danych to: {peaks.ndim}")
        

        if not skip_initialisation:
            self.centroids_initialisation(peaks)

        n_channels,n_peaks = peaks.shape 

        self.U = np.random.rand(self.n_microstates,n_peaks) #inicjalizacja losowej macierzy przynależności
        self.U /= np.sum(self.U,axis=0,keepdims = True) #normalizacja do sumy 1 
        

        #aktualizacja centroidów
        for i in range(self.max_iters):

            previous_U = self.U.copy()
            previous_centroids = self.centroids.copy()

            for m in range(self.n_microstates):
                numerator = np.zeros(n_channels) #licznik (wektor bo z wzoru mnożony przez x_j)
                denominator = 0 #mianownik 
                for j in range(n_peaks):
                    peak = peaks[:,j]
                    u_m_pow = self.U[m,j] ** self.m
                    if np.dot(peak, self.centroids[m,:]) >= 0:
                        numerator += u_m_pow * peak
                    else:
                        numerator += u_m_pow * (-peak)
                    denominator += u_m_pow 
                
                if denominator > 0: #normalizacja
                    new_centroid = numerator / denominator
                    self.centroids[m,:] = new_centroid / np.linalg.norm(new_centroid)


            # aktualizacja macierzy przynależności
            for j in range(n_peaks):
                peak = peaks[:,j]
                distances = np.zeros(self.n_microstates)
                for m_dist in range(self.n_microstates):
                    distance_to_c = np.linalg.norm(peak - self.centroids[m_dist,:])
                    distance_to_neg_c = np.linalg.norm(peak - (-self.centroids[m_dist,:]))
                    distances[m_dist] = min(distance_to_c,distance_to_neg_c)
                
                
                distances = np.maximum(distances, 1e-15)

                
                for m in range(self.n_microstates):
                    ratio_sum = 0
                    for m_2 in range(self.n_microstates):
                        ratio = (distances[m] / distances[m_2]) ** (2 / (self.m - 1))
                        ratio_sum += ratio
                    if ratio_sum > 0:
                        self.U[m,j] = 1 / ratio_sum
                    else:
                        self.U[m,j] = 0
            
            self.U /= np.sum(self.U, axis=0, keepdims=True)

                
            #sprawdzenie zbieżności
            change_U = np.linalg.norm(self.U - previous_U)
            change_centroids = np.linalg.norm(self.centroids - previous_centroids)
            if change_U < self.tol and change_centroids < self.tol:
                break 


    def predict(self,data:np.ndarray):
        if self.algorithm == "kmeans++":
            return self._predict_kmeans(data)
        elif self.algorithm == "fcm":
            return self._predict_fcm(data)
        elif self.algorithm == "fkmeans":
            return self._predict_fcm(data)

    def _predict_kmeans(self,data:np.ndarray):
        """Funkcja, która ma za zadanie przypisać próbki sygnału do etykiet mikrostanów.
        Bazuje na wyuczonych wzorcach mikrostanów i dopasowuje je do próbek sygnału.
        """

        if self.centroids is None or self.labels is None: #obsługa błędu związanego z niewyuczonym modelem
            raise RuntimeError(
                "Model nie został wyuczony, nie można przypisać próbek do etykiet mikrostanów\n"
                f"Zainicjowane centroidy: {self.centroids is None}, etykiety: {self.labels is None}"
            )
        
        if not isinstance(data,np.ndarray): #obsługa błędu związanego z typem danych
            raise TypeError("Błąd: dane nie są tablicą numpy")
        
        if data.ndim == 1 or data.ndim > 3: #obsługa błędu związanego z wymiarami macierzy danych eeg
            raise ValueError(f"Błąd: dane mają nieprawidłowe wymiary. Wymiar danych to: {data.ndim}")

        if data.shape[0] != self.centroids.shape[1]: #obsługa błędu związanego z niezgodnością liczby kanałów
            raise ValueError(
                "Liczba wejściowych kanałów nie odpowiada liczbie kanałów w zainicjowanych centroidach \n"
                f"Liczba kanałów w danych wejściowych: {data.shape[0]} \n" 
                f"liczba kanałów w zainicjowanych centroidach: {self.centroids.shape[1]}"
            )
        
        n_channels,n_samples = data.shape #pobranie wymiarów danych wejściowych (n_kanałów,n_próbek)
        labels = np.zeros(n_samples) #tablica przechowująca etykiety próbek (n_próbek)

        for sample in range(n_samples): #w zakresie n_próbek
            current_sample = data[:,sample] #pobranie danych dla próbki 
            min_distance = np.inf #dla porównań odległości
            
            for m in range(self.n_microstates): #w zakresie n_mikrostanów
                centroid = self.centroids[m,:] #pobranie danych centroidu dla m-tego mikrostanu
                distance_to_c = np.linalg.norm(current_sample - centroid)
                distance_to_neg_c = np.linalg.norm(current_sample - (-centroid))
                distance_to_c_sqrt = distance_to_c ** 2 
                distance_to_neg_c_sqrt = distance_to_neg_c ** 2 
                min_distance_sqrt = min(distance_to_c_sqrt,distance_to_neg_c_sqrt)
                if min_distance_sqrt < min_distance:
                    min_distance = min_distance_sqrt
                    labels[sample] = m 
        return labels 

    def _predict_fcm(self,data:np.ndarray):
        if self.centroids is None: #obsługa błędu związanego z niewyuczonym modelem
            raise RuntimeError(
                "Model nie został wyuczony, nie można przypisać próbek do etykiet mikrostanów\n"
                f"Zainicjowane centroidy: {self.centroids is None}"
            )
        
        if not isinstance(data,np.ndarray): #obsługa błędu związanego z typem danych
            raise TypeError("Błąd: dane nie są tablicą numpy")
        
        if data.ndim == 1 or data.ndim > 3: #obsługa błędu związanego z wymiarami macierzy danych eeg
            raise ValueError(f"Błąd: dane mają nieprawidłowe wymiary. Wymiar danych to: {data.ndim}")
        
        if data.shape[0] != self.centroids.shape[1]: #obsługa błędu związanego z niezgodnością liczby kanałów
            raise ValueError(
                "Liczba wejściowych kanałów nie odpowiada liczbie kanałów w zainicjowanych centroidach \n"
                f"Liczba kanałów w danych wejściowych: {data.shape[0]} \n" 
                f"liczba kanałów w zainicjowanych centroidach: {self.centroids.shape[1]}"
            )
        
        n_channels,n_samples = data.shape #pobranie wymiarów danych wejściowych (n_kanałów,n_próbek)
        U_new = np.zeros((self.n_microstates,n_samples))
            
        for j in range(n_samples):
            peak = data[:,j]
            distances = np.zeros(self.n_microstates)
            for m_dist in range(self.n_microstates):
                distance_to_c = np.linalg.norm(peak - self.centroids[m_dist,:])
                distance_to_neg_c = np.linalg.norm(peak - (-self.centroids[m_dist,:]))
                distances[m_dist] = min(distance_to_c,distance_to_neg_c)

            distances = np.maximum(distances, 1e-15)

            for m in range(self.n_microstates):
                ratio_sum = 0
                for m_2 in range(self.n_microstates):
                    ratio = (distances[m] / distances[m_2]) ** (2 / (self.m - 1))
                    ratio_sum += ratio
                if ratio_sum > 0:
                    U_new[m,j] = 1 / ratio_sum
                else:
                    U_new[m,j] = 0

        U_new /= np.sum(U_new, axis=0, keepdims=True)
                
        return U_new


            





    






        




