import mne

d_path = "/Users/szymbierz/Desktop/notebooks/statystyka/inne/eeg/eeg_files/20241017_kp.bdf"

#kanały zewnętrzne

ch_eog = ["EXG1","EXG2"] #ruchy gałek ocznych
ch_ecg = ["EXG3","EXG4"]

#lista niepotrzebnych kanałów

ch_exclude = [f"EXG{i}" for i in range(5,9)]

#zakresy pasm filtra

bandpass_low = 0.2 #dolny zakres
bandpass_high = 30 #górny zakres
notch = 50 #

#wczytanie surowych danych

raw = mne.io.read_raw_bdf(d_path,
                          preload=True,
                          eog=ch_eog,
                          misc=ch_ecg,
                          exclude=ch_exclude)

#ustawienie montażu elektrod (standard dla sprzętu)

raw.set_montage("biosemi64")

#Filtry
raw.notch_filter(notch)
raw.filter(bandpass_low,bandpass_high)

raw.set_eeg_reference("average") #referencja względem średniej sygnału z elekptrod

raw_ica = raw.copy() #skopiowanie sygnału aby zastosować filtr przed ica

raw_ica.filter(l_freq=1.,h_freq=30.)

ica = mne.preprocessing.ICA(n_components=20,
                            random_state=42,
                            max_iter="auto")
ica.fit(raw_ica)

#sprawdzenie calkowitej wariancji dla kompnentow

var_ratio = ica.get_explained_variance_ratio(raw)

for channel,ratio in var_ratio.items():
    print(f"Niezależne komponenty reprezentują :{ratio*100:.2f}% oryginalnego sygnału.")

#lista przechowująca komponenty do usunięcia

ica.exclude = []

#funkcje do znalezienia artefaktów EOG i ECG

eog_indices,eog_scores = ica.find_bads_eog(raw)
ecg_indices,ecg_scores = ica.find_bads_ecg(raw,ch_name="EXG3")
print("Komponenty EOG", eog_indices)
print("Komponenty ECG", ecg_indices)

#sprawdzenie poprawnosci poprzez analize wariancji

print(ecg_scores)
print(eog_scores)

ica.exclude = [1,0,13,8,14,12,7,9,4]

reconstr_raw = raw.copy()
ica.apply(reconstr_raw)







