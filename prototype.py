import streamlit as st
from pathlib import Path
import pandas as pd
from utils.preprocessing import preprocess_input, load_artifacts, calculate_map

def load_css(file_path: str):
    css_path = Path(file_path)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

@st.cache_resource(show_spinner="Memuat model Machine Learning...")
def get_models_and_scaler():

    return load_artifacts(
        rf_path='model/clf_rf.pkl',
        xgb_path='model/clf_xgb.pkl',
        scaler_path='model/scaler.pkl'
    )

try:
    model_rf, model_xgb, scaler = get_models_and_scaler()
except Exception as e:
    st.error(f"⚠️ Gagal memuat model/scaler: {e}")
    st.stop()


st.markdown('<div class="main-header">🩺 Diabetes Risk Assessment & Decision Support</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Evaluasi risiko diabetes tipe 2 secara komprehensif menggunakan parameter klinis, riwayat medis, dan gaya hidup.</div>',
    unsafe_allow_html=True
)

with st.container(border=True):
    st.markdown('### Cek risiko diabetes Anda dalam 2 menit')
    col_p1, col_p2, col_p3 = st.columns([1.5, 1.5, 4])