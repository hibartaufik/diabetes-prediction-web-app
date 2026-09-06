"""
utils/preprocessing.py
----------------------
Modul untuk memproses data mentah input pengguna dari form web app
menjadi format data yang siap diprediksi oleh model ML (Random Forest & XGBoost).

Pipeline ini mereplikasi persis langkah preprocessing data latih (NHANES):
1. Perhitungan Mean Arterial Pressure (MAP) untuk tekanan darah.
2. Mapping kategori konsumsi alkohol.
3. Binary encoding (Ya/Tidak atau Laki-laki/Perempuan).
4. One-Hot Encoding untuk kategori Ras (dengan drop ras_1 sebagai baseline).
5. Penataan urutan 20 kolom fitur agar identik dengan data latih.
6. Normalisasi menggunakan MinMaxScaler yang sudah di-fit pada data asli.
"""

import pickle
import numpy as np
import pandas as pd

# Urutan 20 kolom fitur final yang diharapkan model & scaler
FEATURE_COLUMNS = [
    'gender',
    'usia',
    'BMI',
    'HbA1c',
    'kadar_kolesterol',
    'alkohol',
    'riw_kolesterol_tinggi',
    'sedih-depresi-putus_asa',
    'gangguan_tidur',
    'gangguan_makan',
    'riw_liver',
    'riw_tiroid',
    'riw_kanker',
    'merokok100',
    'tekanan_darah',
    'ras_2',
    'ras_3',
    'ras_4',
    'ras_6',
    'ras_7'
]

# Mapping kategori alkohol sesuai notebook training
ALKOHOL_MAPPING = {
    0: 0,
    10: 1,
    9: 2,
    8: 3,
    7: 4,
    6: 5,
    5: 6,
    4: 7,
    3: 8,
    2: 9,
    1: 10
}


def calculate_map(sistolik: float, diastolik: float) -> float:
    """
    Menghitung Mean Arterial Pressure (MAP) sebagai representasi tekanan darah.
    Rumus: Diastolik + (Sistolik - Diastolik) / 3
    """
    return diastolik + (sistolik - diastolik) / 3.0


def preprocess_input(raw_input: dict, scaler=None) -> pd.DataFrame:
    """
    Menerima dictionary input mentah dari form Streamlit,
    melakukan transformasi, dan mengembalikan DataFrame 1 baris
    yang sudah dinormalisasi dan memiliki 20 kolom berurutan.

    Parameters:
    -----------
    raw_input : dict
        Dictionary berisi nilai mentah dari user, contoh:
        {
            'gender': 1 (Laki-laki) atau 0 (Perempuan),
            'usia': 45,
            'BMI': 27.5,
            'HbA1c': 5.8,
            'kadar_kolesterol': 190,
            'alkohol_kategori': 0,  # 0 - 10
            'riw_kolesterol_tinggi': 1 (Ya) atau 0 (Tidak),
            'sedih_depresi': 0,     # Skala 0 - 3
            'gangguan_tidur': 1,    # Skala 0 - 3
            'gangguan_makan': 0,    # Skala 0 - 3
            'riw_liver': 0,         # 1 (Ya) atau 0 (Tidak)
            'riw_tiroid': 0,        # 1 (Ya) atau 0 (Tidak)
            'riw_kanker': 0,        # 1 (Ya) atau 0 (Tidak)
            'merokok100': 1,        # 1 (Ya) atau 0 (Tidak)
            'sistolik': 120,        # mmHg
            'diastolik': 80,        # mmHg
            'ras': 3                # Kategori ras: 1, 2, 3, 4, 6, atau 7
        }
    scaler : MinMaxScaler, optional
        Objek scaler yang sudah di-fit. Jika diberikan, data akan di-transform.

    Returns:
    --------
    pd.DataFrame
        DataFrame 1 baris dengan 20 kolom siap inferensi.
    """
    # 1. Hitung Mean Arterial Pressure (MAP)
    tekanan_darah_map = calculate_map(
        float(raw_input['sistolik']),
        float(raw_input['diastolik'])
    )

    # 2. Mapping nilai konsumsi alkohol
    alkohol_mapped = ALKOHOL_MAPPING.get(int(raw_input.get('alkohol_kategori', 0)), 0)

    # 3. One-Hot Encoding kategori Ras (ras_1 sebagai baseline/drop_first)
    ras_selected = int(raw_input.get('ras', 1))
    ras_dummies = {
        'ras_2': 1 if ras_selected == 2 else 0,
        'ras_3': 1 if ras_selected == 3 else 0,
        'ras_4': 1 if ras_selected == 4 else 0,
        'ras_6': 1 if ras_selected == 6 else 0,
        'ras_7': 1 if ras_selected == 7 else 0,
    }

    # 4. Susun dictionary fitur sesuai nama kolom aslinya
    row_data = {
        'gender': int(raw_input['gender']),
        'usia': float(raw_input['usia']),
        'BMI': float(raw_input['BMI']),
        'HbA1c': float(raw_input['HbA1c']),
        'kadar_kolesterol': float(raw_input['kadar_kolesterol']),
        'alkohol': alkohol_mapped,
        'riw_kolesterol_tinggi': int(raw_input['riw_kolesterol_tinggi']),
        'sedih-depresi-putus_asa': int(raw_input['sedih_depresi']),
        'gangguan_tidur': int(raw_input['gangguan_tidur']),
        'gangguan_makan': int(raw_input['gangguan_makan']),
        'riw_liver': int(raw_input['riw_liver']),
        'riw_tiroid': int(raw_input['riw_tiroid']),
        'riw_kanker': int(raw_input['riw_kanker']),
        'merokok100': int(raw_input['merokok100']),
        'tekanan_darah': tekanan_darah_map,
        **ras_dummies
    }

    # 5. Buat DataFrame dan pastikan urutan kolom sesuai FEATURE_COLUMNS
    df = pd.DataFrame([row_data], columns=FEATURE_COLUMNS)

    # 6. Normalisasi menggunakan scaler jika disediakan
    if scaler is not None:
        scaled_array = scaler.transform(df)
        df = pd.DataFrame(scaled_array, columns=FEATURE_COLUMNS)

    return df


def load_artifacts(
    rf_path: str = 'model/clf_rf.pkl',
    xgb_path: str = 'model/clf_xgb.pkl',
    scaler_path: str = 'model/scaler.pkl'
):
    """
    Helper function untuk memuat kedua model dan scaler dari disk.
    """
    with open(rf_path, 'rb') as f:
        model_rf = pickle.load(f)
    with open(xgb_path, 'rb') as f:
        model_xgb = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model_rf, model_xgb, scaler
