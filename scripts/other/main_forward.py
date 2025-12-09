import mne
import pandas as pd


raw_path = "/Users/szymbierz/Desktop/notebooks/statystyka/eeg/eeg_files/20241017_kp_cleaned.edf"
raw = mne.io.read_raw_edf(raw_path,
                          preload=True,exclude=["EXG1","EXG2","EXG3","EXG4","EXG5"])

raw.set_montage("biosemi64")

"""Uproszczony model głowy"""
sphere_model = mne.make_sphere_model(info=raw.info)

"""Tworzenie siatki dipoli"""
src = mne.setup_volume_source_space(subject=None,pos=3)

"""Rozwiązanie forward problem"""

fwd = mne.make_forward_solution(raw.info,
                                eeg=True,
                                meg=False,
                                trans=None,
                                src=src,
                                bem=sphere_model)

"""W celu uproszczenia"""

fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)


macierz_zyskow = fwd_fixed['sol']['data'] # macierz o wymiarach (n_channels, n_dipoles*3), gdzie wiersze to elektrody
df = pd.DataFrame(macierz_zyskow)
print(df)