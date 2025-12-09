# 🧠 EEG Microstates Analysis

Repozytorium zawierające skrypty i notebooki do analizy mikrostanów EEG.

## 📁 Struktura projektu

```
eeg/
├── scripts/
│   ├── microstates/          # Główne skrypty do analizy mikrostanów
│   │   ├── class_microstates.py      # Klasa Microstates (k-means++, FCM)
│   │   ├── class_MicrostateMetrics.py # Metryki mikrostanów
│   │   ├── preprocessing.py          # Preprocessing z MNE
│   │   ├── epochs_transforms.py      # Transformacje epok
│   │   ├── count_gfp_find_peaks.py   # Obliczanie GFP i peak detection
│   │   ├── visualise_microstates.py  # Wizualizacje
│   │   └── ...
│   └── other/                # Dodatkowe skrypty (inverse, forward, qEEG)
├── Notebooks/                # Notebooki Jupyter z analizami
├── Dane/
│   ├── raw/                  # Surowe pliki EEG (NIE w repo - za duże)
│   └── processed/            # Przetworzone dane (małe pliki w repo)
└── requirements.txt          # Zależności Python
```

## 🚀 Instalacja

### 1. Klonowanie repozytorium

#### Przez SSH (zalecane):
```bash
git clone git@github.com:szymbierz/eeg_microstates.git
cd eeg_microstates
```

#### Przez HTTPS (alternatywa):
```bash
git clone https://github.com/szymbierz/eeg_microstates.git
cd eeg_microstates
```

### 2. Konfiguracja środowiska

#### Windows z Conda (używając istniejącego środowiska)

Jeśli masz już środowisko conda w folderze `projekty_naukowe`:

```powershell
# Aktywuj swoje istniejące środowisko conda
conda activate nazwa_twojego_srodowiska

# Przejdź do sklonowanego folderu
cd C:\ścieżka\do\eeg_microstates

# Zainstaluj zależności w istniejącym środowisku
pip install -r requirements.txt
```

**Uwaga:** Jeśli nie masz jeszcze środowiska conda, możesz je utworzyć:
```powershell
conda create -n projekty_naukowe python=3.11
conda activate projekty_naukowe
cd C:\ścieżka\do\eeg_microstates
pip install -r requirements.txt
```

#### macOS / Linux (venv)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Windows (venv - alternatywa)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 📦 Przenoszenie dużych plików danych

⚠️ **Pliki EEG (`.bdf`, `.edf`, `.fif` > 100MB) NIE są w repozytorium GitHub!**

### Pliki do ręcznego przeniesienia:

| Plik | Rozmiar | Lokalizacja |
|------|---------|-------------|
| `20241017_kp.bdf` | 612 MB | `Dane/raw/` |
| `main_pyprep.fif` | 771 MB | `Dane/processed/` |
| `epochs.edf` | 630 MB | `Dane/processed/` |
| `epochs_ar_1.set` | 630 MB | `Dane/processed/` |
| `20241017_kp_cleaned.edf` | 385 MB | `Dane/processed/` |

### Jak przenieść dane:

1. **Pendrive/Dysk zewnętrzny** - skopiuj folder `Dane/` z dużymi plikami
2. **Dysk sieciowy / NAS** - jeśli masz dostęp
3. **Cloud storage** - Google Drive, OneDrive, Dropbox
4. **Git LFS** - dla zaawansowanych (wymaga konfiguracji)

## 🔧 Konfiguracja ścieżek

Po sklonowaniu na nowy komputer, zaktualizuj ścieżki w skryptach:

### macOS (oryginał)
```python
d_path = "/Users/szymbierz/Desktop/notebooks/statystyka/inne/eeg/Dane/raw/20241017_kp.bdf"
```

### Windows (nowy komputer)
```python
d_path = r"C:\Users\TwojaNamea\Documents\eeg-microstates\Dane\raw\20241017_kp.bdf"
# lub użyj pathlib:
from pathlib import Path
d_path = Path(__file__).parent.parent / "Dane" / "raw" / "20241017_kp.bdf"
```

### 💡 Tip: Użyj zmiennych środowiskowych lub `config.py`

Stwórz plik `config.py` (nie commituj go):
```python
# config.py
from pathlib import Path
import os

# Automatycznie wykryj system
if os.name == 'nt':  # Windows
    DATA_DIR = Path(r"C:\Users\TwojaNamea\Documents\eeg-microstates\Dane")
else:  # macOS/Linux
    DATA_DIR = Path.home() / "Desktop/wszystko/notebooks/statystyka/inne/eeg/Dane"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
```

## 📚 Main dependencies 

- **MNE-Python** 
- **NumPy** 
- **SciPy** 
- **Matplotlib** 


## 🎓 Usage

### Microstates Class

```python
from scripts.microstates.class_microstates import Microstates
import numpy as np

# peaks: (n_channels, n_peaks) - mapy topograficzne z GFP peaks
microstates = Microstates(
    peaks=peaks_data,
    n_microstates=4,
    max_iters=100,
    algorithm="kmeans++"  # lub "fcm", "fkmeans"
)

microstates.fit(peaks_data)
labels = microstates.predict(eeg_data)
```

## 📝 Notes

- Projekt pisany na **macOS**, testowany na **Windows**
- Używaj `pathlib.Path` zamiast stringów dla cross-platform kompatybilności
- Pamiętaj o aktualizacji montażu elektrod jeśli używasz innego sprzętu

## 👨‍🔬 Author

Szymon Bierzanowski

## 📄 License
MIT

