# 📋 Instrukcja instalacji na Windows z Conda

Krótki przewodnik jak pobrać i skonfigurować projekt na Windows używając istniejącego środowiska conda.

## Krok 1: Klonowanie repozytorium

### Opcja A: Przez SSH (zalecane - jeśli masz skonfigurowany SSH)

Otwórz **Git Bash** lub **PowerShell** i wykonaj:

```bash
# Przejdź do folderu gdzie chcesz mieć projekt (np. projekty_naukowe)
cd C:\Users\TwojaNazwa\projekty_naukowe

# Sklonuj repozytorium
git clone git@github.com:szymbierz/eeg_microstates.git

# Przejdź do folderu projektu
cd eeg_microstates
```

### Opcja B: Przez HTTPS

```bash
cd C:\Users\TwojaNazwa\projekty_naukowe
git clone https://github.com/szymbierz/eeg_microstates.git
cd eeg_microstates
```

## Krok 2: Aktywacja istniejącego środowiska Conda

```powershell
# Aktywuj swoje istniejące środowisko conda
conda activate nazwa_twojego_srodowiska

# Sprawdź czy jesteś w odpowiednim folderze
pwd  # lub w PowerShell: Get-Location
```

## Krok 3: Instalacja zależności

```powershell
# Upewnij się, że jesteś w folderze eeg_microstates
cd C:\Users\TwojaNazwa\projekty_naukowe\eeg_microstates

# Zainstaluj wszystkie wymagane pakiety
pip install -r requirements.txt
```

## Krok 4: Przeniesienie dużych plików danych

⚠️ **WAŻNE:** Duże pliki EEG nie są w repozytorium!

1. Skopiuj folder `Dane/` z Maca (przez pendrive/cloud)
2. Wklej go do `C:\Users\TwojaNazwa\projekty_naukowe\eeg_microstates\Dane\`

Struktura powinna wyglądać tak:
```
eeg_microstates/
├── Dane/
│   ├── raw/
│   │   └── 20241017_kp.bdf  (612 MB)
│   └── processed/
│       ├── main_pyprep.fif  (771 MB)
│       ├── epochs.edf       (630 MB)
│       └── ...
```

## Krok 5: Konfiguracja ścieżek w skryptach

### Automatyczna konfiguracja (zalecane)

Stwórz plik `config.py` w głównym folderze projektu:

```python
# config.py
from pathlib import Path
import os

# Automatycznie wykryj lokalizację projektu
PROJECT_ROOT = Path(__file__).parent

# Ścieżki do danych
DATA_DIR = PROJECT_ROOT / "Dane"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Przykład użycia:
# d_path = RAW_DIR / "20241017_kp.bdf"
```

### Ręczna aktualizacja ścieżek

W plikach takich jak `scripts/microstates/preprocessing.py` znajdź:

```python
# macOS (stara ścieżka):
d_path = "/Users/szymbierz/Desktop/notebooks/statystyka/inne/eeg/Dane/raw/20241017_kp.bdf"
```

I zamień na:

```python
# Windows (nowa ścieżka):
from pathlib import Path
d_path = Path(r"C:\Users\TwojaNazwa\projekty_naukowe\eeg_microstates\Dane\raw\20241017_kp.bdf")
```

Lub jeszcze lepiej - użyj `config.py`:

```python
from config import RAW_DIR
d_path = RAW_DIR / "20241017_kp.bdf"
```

## Krok 6: Testowanie

Otwórz Jupyter Notebook:

```powershell
# Upewnij się, że środowisko conda jest aktywne
conda activate nazwa_twojego_srodowiska

# Uruchom Jupyter
jupyter notebook
```

Lub JupyterLab:

```powershell
jupyter lab
```

## 🔧 Rozwiązywanie problemów

### Problem: "git clone" nie działa przez SSH

**Rozwiązanie:** Użyj HTTPS lub skonfiguruj SSH na Windows:
1. Wygeneruj klucz SSH: `ssh-keygen -t ed25519 -C "twoj@email.com"`
2. Dodaj klucz do GitHub: https://github.com/settings/keys

### Problem: "conda: command not found"

**Rozwiązanie:** Zainstaluj Anaconda/Miniconda lub dodaj conda do PATH.

### Problem: Ścieżki w notebookach nie działają

**Rozwiązanie:** 
- Użyj `pathlib.Path` zamiast stringów
- Upewnij się, że folder `Dane/` jest w odpowiednim miejscu
- Sprawdź czy duże pliki zostały skopiowane

### Problem: Brakujące moduły (np. mne)

**Rozwiązanie:**
```powershell
conda activate nazwa_twojego_srodowiska
pip install mne numpy scipy matplotlib
```

## ✅ Checklist

- [ ] Repozytorium sklonowane
- [ ] Środowisko conda aktywowane
- [ ] Zależności zainstalowane (`pip install -r requirements.txt`)
- [ ] Folder `Dane/` z dużymi plikami skopiowany
- [ ] Ścieżki w skryptach zaktualizowane (lub `config.py` utworzony)
- [ ] Jupyter Notebook działa
- [ ] Test importu: `from scripts.microstates.class_microstates import Microstates`

## 📞 Pomoc

Jeśli masz problemy, sprawdź:
- Czy wszystkie pliki z `Dane/` zostały skopiowane
- Czy ścieżki są poprawne (Windows używa `\` lub `Path`)
- Czy środowisko conda ma wszystkie pakiety

