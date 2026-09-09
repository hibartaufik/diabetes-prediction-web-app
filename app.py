"""
app.py
------
Aplikasi Web Prediksi Risiko Diabetes dengan Streamlit.
Desain dan alur UX diadaptasi 1:1 dari prototipe klinis (diabetes-risk-prototype.html):
- Multi-step wizard (Landing -> Demografis -> Medis -> Mental -> Gaya Hidup -> Hasil)
- Tipografi Source Serif 4 & Inter
- Visualisasi status risiko, dual-model (Screening vs Diagnostic), factor bars, dan rekomendasi.
"""

from pathlib import Path
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
from utils.preprocessing import preprocess_input, load_artifacts, calculate_map

# ==========================================
# 1. Konfigurasi Halaman & Styling
# ==========================================
st.set_page_config(
    page_title="Cek Risiko Diabetes — Skrining Mandiri",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Muat style.css
css_file = Path("style.css")
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ==========================================
# 2. Pemuatan Model & Scaler
# ==========================================
@st.cache_resource(show_spinner="Menyiapkan sistem skrining...")
def get_artifacts():
    return load_artifacts(
        rf_path='model/clf_rf.pkl',
        xgb_path='model/clf_xgb.pkl',
        scaler_path='model/scaler.pkl'
    )

try:
    model_rf, model_xgb, scaler = get_artifacts()
except Exception as e:
    st.error(f"⚠️ Gagal memuat model: {e}")
    st.stop()


# ==========================================
# 3. Manajemen State Form & Navigasi Layar
# ==========================================
if "step" not in st.session_state:
    st.session_state.step = "landing"

# Inisialisasi default form jika belum ada
if "form_data" not in st.session_state:
    st.session_state.form_data = {
        "gender": "Perempuan",
        "usia": 34,
        "ras": "Asia (termasuk Indonesia & Asia Tenggara)",
        "rw_kol": "Tidak",
        "rw_liv": "Tidak",
        "rw_tir": "Tidak",
        "rw_kan": "Tidak",
        "sistolik": 120,
        "diastolik": 80,
        "bmi": 24.5,
        "hba1c": 5.4,
        "kolesterol": 185.0,
        "depresi": "Tidak pernah",
        "tidur": "Cukup & teratur",
        "makan": "Teratur & seimbang",
        "alkohol": "Tidak pernah",
        "rokok": "Tidak pernah"
    }

def go_to(step_name: str):
    st.session_state.step = step_name
    st.rerun()

def render_topbar():
    st.markdown("""
    <div class="topbar">
      <div class="mark">+</div>
      <div>
        <div class="name">Cek Diabetes</div>
        <div class="tagline">Skrining risiko mandiri</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_progress(step_num: int, label: str):
    segs = []
    for i in range(1, 5):
        if i < step_num:
            segs.append('<div class="progress-seg done"><div class="fill"></div></div>')
        elif i == step_num:
            segs.append('<div class="progress-seg current"><div class="fill"></div></div>')
        else:
            segs.append('<div class="progress-seg"><div class="fill"></div></div>')
    
    html = f"""
    <div class="progress-wrap">
      <div class="progress-label"><span>Langkah {step_num} dari 4</span><span>{label}</span></div>
      <div class="progress-track">
        {''.join(segs)}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# Tampilkan Top Bar di semua layar
render_topbar()


# =========================================================
# SCREEN 1: LANDING SCREEN
# =========================================================
if st.session_state.step == "landing":
    st.markdown("""
    <div class="hero-card">
      <div class="hero-icon">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#3E8E7E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
        </svg>
      </div>
      <h1>Cek risiko diabetes Anda dalam 2 menit</h1>
      <p>Jawab beberapa pertanyaan tentang kondisi tubuh, kebiasaan, dan keseharian Anda. Kami bantu memberi gambaran awal risiko diabetes secara pribadi dan mudah dipahami.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Mulai pemeriksaan", type="primary", use_container_width=True):
        go_to("form1")

    st.markdown("""
    <div class="trust-row">
      <div class="trust-item">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64798A" stroke-width="2">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        </svg>
        Data tidak disimpan
      </div>
      <div class="trust-item">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64798A" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <path d="M12 6v6l4 2"/>
        </svg>
        Hasil instan
      </div>
    </div>
    <div class="disclaimer-box">
      <span>ⓘ</span>
      <span>Hasil pemeriksaan ini adalah gambaran risiko awal, bukan diagnosis medis. Untuk kepastian, konsultasikan dengan tenaga kesehatan.</span>
    </div>
    """, unsafe_allow_html=True)

    # Preset Shortcut untuk Pengujian Cepat
    st.write("")
    with st.expander("⚡ Coba Contoh Data Cepat (Testing Preset)"):
        col_ps1, col_ps2 = st.columns(2)
        if col_ps1.button("🟢 Profil Sehat / Rendah", use_container_width=True):
            st.session_state.form_data.update({
                "gender": "Perempuan", "usia": 28, "ras": "Asia (termasuk Indonesia & Asia Tenggara)",
                "rw_kol": "Tidak", "rw_liv": "Tidak", "rw_tir": "Tidak", "rw_kan": "Tidak",
                "sistolik": 115, "diastolik": 75, "bmi": 21.8, "hba1c": 5.0, "kolesterol": 170.0,
                "depresi": "Tidak pernah", "tidur": "Cukup & teratur", "makan": "Teratur & seimbang",
                "alkohol": "Tidak pernah", "rokok": "Tidak pernah"
            })
            go_to("result")
        if col_ps2.button("🔴 Profil Risiko Tinggi", use_container_width=True):
            st.session_state.form_data.update({
                "gender": "Laki-laki", "usia": 56, "ras": "Hispanik / Latino",
                "rw_kol": "Ya", "rw_liv": "Tidak", "rw_tir": "Ya", "rw_kan": "Tidak",
                "sistolik": 145, "diastolik": 92, "bmi": 32.5, "hba1c": 7.4, "kolesterol": 238.0,
                "depresi": "Sering", "tidur": "Hampir setiap hari terganggu", "makan": "Hampir setiap hari bermasalah",
                "alkohol": "Hampir setiap hari", "rokok": "Perokok aktif"
            })
            go_to("result")


# =========================================================
# SCREEN 2: FORM 1 - DEMOGRAFIS & RIWAYAT MEDIS
# =========================================================
elif st.session_state.step == "form1":
    render_progress(1, "Demografis")

    st.markdown("""
    <div class="form-card">
      <div class="section-eyebrow">Tentang Anda</div>
      <h2>Data Demografis</h2>
      <p class="section-desc">Informasi dasar ini membantu menyesuaikan hasil pemeriksaan dengan profil Anda.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Jenis Kelamin**")
        gender_choice = st.pills(
            "Jenis Kelamin",
            options=["Perempuan", "Laki-laki"],
            default=st.session_state.form_data["gender"],
            label_visibility="collapsed"
        ) or st.session_state.form_data["gender"]
        st.session_state.form_data["gender"] = gender_choice

        st.write("")
        usia_input = st.number_input(
            "Usia (tahun)",
            min_value=20, max_value=85,
            value=int(st.session_state.form_data["usia"]),
            step=1
        )
        st.session_state.form_data["usia"] = usia_input

        ras_list = [
            "Asia (termasuk Indonesia & Asia Tenggara)",
            "Kaukasia / Kulit Putih",
            "Afrika / Kulit Hitam",
            "Hispanik / Latino",
            "Mexican American",
            "Lainnya / Etnis Campuran"
        ]
        cur_ras = st.session_state.form_data.get("ras", "Asia (termasuk Indonesia & Asia Tenggara)")
        ras_idx = ras_list.index(cur_ras) if cur_ras in ras_list else 0
        ras_choice = st.selectbox("Ras / etnisitas", options=ras_list, index=ras_idx)
        st.session_state.form_data["ras"] = ras_choice

        st.markdown("---")
        st.markdown("**Riwayat Medis yang Pernah Didiagnosis Dokter:**")
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.caption("Kolesterol Tinggi")
            st.session_state.form_data["rw_kol"] = st.pills(
                "Kolesterol Tinggi", ["Tidak", "Ya"],
                default=st.session_state.form_data["rw_kol"],
                label_visibility="collapsed", key="p_rw_kol"
            ) or st.session_state.form_data["rw_kol"]

            st.caption("Masalah Liver / Hati")
            st.session_state.form_data["rw_liv"] = st.pills(
                "Liver", ["Tidak", "Ya"],
                default=st.session_state.form_data["rw_liv"],
                label_visibility="collapsed", key="p_rw_liv"
            ) or st.session_state.form_data["rw_liv"]

        with c_m2:
            st.caption("Gangguan Tiroid")
            st.session_state.form_data["rw_tir"] = st.pills(
                "Tiroid", ["Tidak", "Ya"],
                default=st.session_state.form_data["rw_tir"],
                label_visibility="collapsed", key="p_rw_tir"
            ) or st.session_state.form_data["rw_tir"]

            st.caption("Riwayat Kanker")
            st.session_state.form_data["rw_kan"] = st.pills(
                "Kanker", ["Tidak", "Ya"],
                default=st.session_state.form_data["rw_kan"],
                label_visibility="collapsed", key="p_rw_kan"
            ) or st.session_state.form_data["rw_kan"]

    st.write("")
    c_nav1, c_nav2 = st.columns([1, 2])
    with c_nav1:
        if st.button("Kembali", type="secondary", use_container_width=True):
            go_to("landing")
    with c_nav2:
        if st.button("Lanjut", type="primary", use_container_width=True):
            go_to("form2")


# =========================================================
# SCREEN 3: FORM 2 - PEMERIKSAAN MEDIS
# =========================================================
elif st.session_state.step == "form2":
    render_progress(2, "Pemeriksaan Medis")

    st.markdown("""
    <div class="form-card">
      <div class="section-eyebrow">Angka dari Pemeriksaan</div>
      <h2>Data Medis</h2>
      <p class="section-desc">Gunakan hasil pemeriksaan terakhir Anda. Perkiraan juga tidak masalah.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Tekanan darah (mmHg)**")
        bp1, bp2 = st.columns(2)
        with bp1:
            sis_val = st.number_input("Sistolik", min_value=80, max_value=220, value=int(st.session_state.form_data["sistolik"]), step=1)
            st.session_state.form_data["sistolik"] = sis_val
        with bp2:
            dia_val = st.number_input("Diastolik", min_value=50, max_value=140, value=int(st.session_state.form_data["diastolik"]), step=1)
            st.session_state.form_data["diastolik"] = dia_val

        st.write("")
        bmi_val = st.number_input(
            "BMI (indeks massa tubuh)",
            min_value=15.0, max_value=70.0,
            value=float(st.session_state.form_data["bmi"]),
            step=0.1,
            help="Berat badan (kg) dibagi kuadrat tinggi badan (m)"
        )
        st.session_state.form_data["bmi"] = bmi_val

        hba1c_val = st.number_input(
            "HbA1C (%)",
            min_value=3.5, max_value=16.0,
            value=float(st.session_state.form_data["hba1c"]),
            step=0.1,
            help="Kadar gula darah rata-rata 3 bulan terakhir. Normal < 5.7%, Prediabetes 5.7-6.4%, Diabetes ≥ 6.5%"
        )
        st.session_state.form_data["hba1c"] = hba1c_val

        kol_val = st.number_input(
            "Kadar kolesterol total (mg/dL)",
            min_value=60.0, max_value=450.0,
            value=float(st.session_state.form_data["kolesterol"]),
            step=1.0,
            help="Normal: < 200 mg/dL"
        )
        st.session_state.form_data["kolesterol"] = kol_val

    st.write("")
    c_nav1, c_nav2 = st.columns([1, 2])
    with c_nav1:
        if st.button("Kembali", type="secondary", use_container_width=True):
            go_to("form1")
    with c_nav2:
        if st.button("Lanjut", type="primary", use_container_width=True):
            go_to("form3")


# =========================================================
# SCREEN 4: FORM 3 - KESEHATAN MENTAL
# =========================================================
elif st.session_state.step == "form3":
    render_progress(3, "Kesehatan Mental")

    st.markdown("""
    <div class="form-card">
      <div class="section-eyebrow">Keseharian Anda</div>
      <h2>Kesehatan Mental</h2>
      <p class="section-desc">Kondisi mental turut memengaruhi risiko kesehatan tubuh secara keseluruhan.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Seberapa sering merasa sedih, tertekan, atau putus asa dalam 2 minggu terakhir?**")
        scale_opts = ["Tidak pernah", "Kadang", "Sering", "Hampir selalu"]
        dep_val = st.segmented_control(
            "Depresi",
            options=scale_opts,
            default=st.session_state.form_data["depresi"],
            label_visibility="collapsed"
        ) or st.session_state.form_data["depresi"]
        st.session_state.form_data["depresi"] = dep_val

        st.write("")
        st.markdown("**Bagaimana pola tidur Anda belakangan ini?**")
        tidur_opts = [
            "Cukup & teratur",
            "Kadang terganggu",
            "Sering kurang tidur",
            "Hampir setiap hari terganggu"
        ]
        tidur_val = st.pills(
            "Pola Tidur",
            options=tidur_opts,
            default=st.session_state.form_data["tidur"],
            label_visibility="collapsed"
        ) or st.session_state.form_data["tidur"]
        st.session_state.form_data["tidur"] = tidur_val

        st.write("")
        st.markdown("**Bagaimana pola makan Anda belakangan ini?**")
        makan_opts = [
            "Teratur & seimbang",
            "Kadang tidak teratur",
            "Sering berlebihan/kurang",
            "Hampir setiap hari bermasalah"
        ]
        makan_val = st.pills(
            "Pola Makan",
            options=makan_opts,
            default=st.session_state.form_data["makan"],
            label_visibility="collapsed"
        ) or st.session_state.form_data["makan"]
        st.session_state.form_data["makan"] = makan_val

    st.write("")
    c_nav1, c_nav2 = st.columns([1, 2])
    with c_nav1:
        if st.button("Kembali", type="secondary", use_container_width=True):
            go_to("form2")
    with c_nav2:
        if st.button("Lanjut", type="primary", use_container_width=True):
            go_to("form4")


# =========================================================
# SCREEN 5: FORM 4 - GAYA HIDUP
# =========================================================
elif st.session_state.step == "form4":
    render_progress(4, "Gaya Hidup")

    st.markdown("""
    <div class="form-card">
      <div class="section-eyebrow">Langkah Terakhir</div>
      <h2>Gaya Hidup</h2>
      <p class="section-desc">Kebiasaan ini membantu melengkapi gambaran risiko Anda secara menyeluruh.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Konsumsi minuman beralkohol**")
        alk_opts = [
            "Tidak pernah",
            "Jarang (beberapa kali setahun)",
            "Bulanan (1-3 kali sebulan)",
            "Mingguan (1-4 kali seminggu)",
            "Hampir setiap hari"
        ]
        alk_val = st.pills(
            "Alkohol",
            options=alk_opts,
            default=st.session_state.form_data["alkohol"],
            label_visibility="collapsed"
        ) or st.session_state.form_data["alkohol"]
        st.session_state.form_data["alkohol"] = alk_val

        st.write("")
        st.markdown("**Status merokok**")
        rokok_opts = ["Tidak pernah", "Mantan perokok", "Perokok aktif"]
        rokok_val = st.pills(
            "Merokok",
            options=rokok_opts,
            default=st.session_state.form_data["rokok"],
            label_visibility="collapsed"
        ) or st.session_state.form_data["rokok"]
        st.session_state.form_data["rokok"] = rokok_val

    st.write("")
    c_nav1, c_nav2 = st.columns([1, 2])
    with c_nav1:
        if st.button("Kembali", type="secondary", use_container_width=True):
            go_to("form3")
    with c_nav2:
        if st.button("Lihat hasil saya", type="primary", use_container_width=True):
            go_to("result")


# =========================================================
# SCREEN 6: RESULT SCREEN (HASIL PENILAIAN DUAL-MODEL)
# =========================================================
elif st.session_state.step == "result":
    # 1. Mapping input ke nilai numerik untuk model
    fd = st.session_state.form_data

    gender_code = 1 if fd["gender"] == "Laki-laki" else 0

    ras_map = {
        "Mexican American": 1,
        "Hispanik / Latino": 2,
        "Kaukasia / Kulit Putih": 3,
        "Afrika / Kulit Hitam": 4,
        "Asia (termasuk Indonesia & Asia Tenggara)": 6,
        "Lainnya / Etnis Campuran": 7
    }
    ras_code = ras_map.get(fd["ras"], 6)

    rw_kol_code = 1 if fd["rw_kol"] == "Ya" else 0
    rw_liv_code = 1 if fd["rw_liv"] == "Ya" else 0
    rw_tir_code = 1 if fd["rw_tir"] == "Ya" else 0
    rw_kan_code = 1 if fd["rw_kan"] == "Ya" else 0

    depresi_map = {"Tidak pernah": 0, "Kadang": 1, "Sering": 2, "Hampir selalu": 3}
    depresi_code = depresi_map.get(fd["depresi"], 0)

    tidur_map = {
        "Cukup & teratur": 0,
        "Kadang terganggu": 1,
        "Sering kurang tidur": 2,
        "Hampir setiap hari terganggu": 3
    }
    tidur_code = tidur_map.get(fd["tidur"], 0)

    makan_map = {
        "Teratur & seimbang": 0,
        "Kadang tidak teratur": 1,
        "Sering berlebihan/kurang": 2,
        "Hampir setiap hari bermasalah": 3
    }
    makan_code = makan_map.get(fd["makan"], 0)

    # Pemetaan ke kode mentah NHANES untuk alkohol:
    # 0 -> mapped: 0 (tidak pernah)
    # 9 -> mapped: 2 (jarang / beberapa kali setahun)
    # 7 -> mapped: 4 (bulanan / 1-3x sebulan)
    # 5 -> mapped: 6 (mingguan / 1-4x seminggu)
    # 1 -> mapped: 10 (hampir setiap hari / harian)
    alkohol_map = {
        "Tidak pernah": 0,
        "Jarang (beberapa kali setahun)": 9,
        "Bulanan (1-3 kali sebulan)": 7,
        "Mingguan (1-4 kali seminggu)": 5,
        "Hampir setiap hari": 1
    }
    alkohol_code = alkohol_map.get(fd["alkohol"], 0)

    # Merokok minimal 100 batang seumur hidup (mantan/aktif = 1)
    rokok_code = 1 if fd["rokok"] in ["Mantan perokok", "Perokok aktif"] else 0

    # 2. Susun dictionary raw input
    raw_dict = {
        'gender': gender_code,
        'usia': float(fd["usia"]),
        'BMI': float(fd["bmi"]),
        'HbA1c': float(fd["hba1c"]),
        'kadar_kolesterol': float(fd["kolesterol"]),
        'alkohol_kategori': alkohol_code,
        'riw_kolesterol_tinggi': rw_kol_code,
        'sedih_depresi': depresi_code,
        'gangguan_tidur': tidur_code,
        'gangguan_makan': makan_code,
        'riw_liver': rw_liv_code,
        'riw_tiroid': rw_tir_code,
        'riw_kanker': rw_kan_code,
        'merokok100': rokok_code,
        'sistolik': float(fd["sistolik"]),
        'diastolik': float(fd["diastolik"]),
        'ras': ras_code
    }

    # Transformasi & Prediksi
    input_df = preprocess_input(raw_dict, scaler=scaler)
    rf_prob = float(model_rf.predict_proba(input_df)[0][1])
    xgb_prob = float(model_xgb.predict_proba(input_df)[0][1])

    # Penentuan Kategori Risiko
    # Mode Screening (RF) sensitif mendeteksi risiko
    if xgb_prob >= 0.50 or rf_prob >= 0.65:
        badge_cls = "high"
        badge_text = "Risiko Tinggi"
        heading_text = "Ada beberapa hal yang perlu Anda perhatikan"
        sub_text = "Berdasarkan data yang Anda masukkan, beberapa indikator klinis menunjukkan tanda risiko diabetes yang signifikan."
    elif rf_prob >= 0.35 or xgb_prob >= 0.25:
        badge_cls = "moderate"
        badge_text = "Risiko Sedang"
        heading_text = "Ada beberapa faktor yang perlu diwaspadai"
        sub_text = "Model skrining awal menangkap potensi indikasi risiko diabetes yang sebaiknya dievaluasi lebih lanjut."
    else:
        badge_cls = "low"
        badge_text = "Risiko Rendah"
        heading_text = "Profil risiko Anda saat ini tergolong rendah"
        sub_text = "Berdasarkan data yang Anda masukkan, indikator tubuh dan kebiasaan Anda berada dalam rentang yang aman."

    # 1. Result Badge Card
    st.markdown(f"""
    <div class="result-badge">
      <div class="result-status {badge_cls}"><span class="dot"></span> {badge_text}</div>
      <h1>{heading_text}</h1>
      <p class="result-sub">{sub_text}</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Dual-Model Estimation Card
    st.markdown(f"""
    <div class="dual-model-card">
      <div style="font-size: 13px; font-weight: 600; color: #1D4E6B; margin-bottom: 10px;">
        ESTIMASI PROBABILITAS (DUAL-MODEL AI)
      </div>
      <div class="dual-model-box">
        <div class="mini-model-card">
          <div class="mini-model-title">🩺 Random Forest</div>
          <div class="mini-model-mode">Skrining Awal (Recall Tinggi)</div>
          <div class="mini-model-prob">{rf_prob:.1%}</div>
        </div>
        <div class="mini-model-card">
          <div class="mini-model-title">⚖️ XGBoost</div>
          <div class="mini-model-mode">Konfirmasi (Akurasi F1 Seimbang)</div>
          <div class="mini-model-prob">{xgb_prob:.1%}</div>
        </div>
      </div>
      <div style="font-size: 12px; color: #64798A; line-height: 1.45;">
        <b>Catatan:</b> Random Forest difokuskan untuk menangkap risiko secara sensitif (meminimalkan risiko yang terlewat), sedangkan XGBoost memberikan kepastian pola diagnosis yang seimbang.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Faktor yang Memengaruhi Hasil (Factor Bars)
    # Hitung level keparahan dinamis
    # HbA1c
    hba1c_val = fd["hba1c"]
    if hba1c_val >= 6.5:
        hba1c_cls, hba1c_lvl, hba1c_pct = "high", "Di atas batas normal (Diabetes)", 85
    elif hba1c_val >= 5.7:
        hba1c_cls, hba1c_lvl, hba1c_pct = "mid", "Perlu perhatian (Prediabetes)", 60
    else:
        hba1c_cls, hba1c_lvl, hba1c_pct = "ok", "Optimal (< 5.7%)", 25

    # BMI
    bmi_val = fd["bmi"]
    if bmi_val >= 30:
        bmi_cls, bmi_lvl, bmi_pct = "high", "Obesitas (≥ 30)", 78
    elif bmi_val >= 25:
        bmi_cls, bmi_lvl, bmi_pct = "mid", "Overweight (25 - 29.9)", 55
    else:
        bmi_cls, bmi_lvl, bmi_pct = "ok", "Ideal (Normal)", 25

    # Tekanan Darah
    sis_v, dia_v = fd["sistolik"], fd["diastolik"]
    map_v = calculate_map(sis_v, dia_v)
    if sis_v >= 140 or dia_v >= 90:
        bp_cls, bp_lvl, bp_pct = "high", f"Tinggi ({sis_v}/{dia_v} mmHg)", 70
    elif sis_v >= 130 or dia_v >= 85:
        bp_cls, bp_lvl, bp_pct = "mid", f"Pre-hipertensi ({sis_v}/{dia_v})", 45
    else:
        bp_cls, bp_lvl, bp_pct = "ok", f"Normal ({sis_v}/{dia_v} mmHg)", 20

    # Pola Tidur & Keseharian
    if fd["tidur"] == "Hampir setiap hari terganggu":
        tidur_cls, tidur_lvl, tidur_pct = "high", "Sangat terganggu (Harian)", 85
    elif fd["tidur"] == "Sering kurang tidur":
        tidur_cls, tidur_lvl, tidur_pct = "mid", "Sering terganggu", 60
    elif fd["tidur"] == "Kadang terganggu":
        tidur_cls, tidur_lvl, tidur_pct = "mid", "Kadang terganggu", 35
    else:
        tidur_cls, tidur_lvl, tidur_pct = "ok", "Cukup & teratur", 20

    st.markdown(f"""
    <div class="factors-card">
      <h3>Apa yang memengaruhi hasil ini</h3>
      <p class="factors-desc">Urutan faktor berdasarkan kontribusinya terhadap profil risiko Anda</p>

      <div class="factor-row">
        <div class="factor-top"><span class="fname">Kadar HbA1C ({hba1c_val}%)</span><span class="flevel">{hba1c_lvl}</span></div>
        <div class="factor-bar"><div class="fill {hba1c_cls}" style="width: {hba1c_pct}%"></div></div>
      </div>
      <div class="factor-row">
        <div class="factor-top"><span class="fname">Indeks massa tubuh (BMI {bmi_val})</span><span class="flevel">{bmi_lvl}</span></div>
        <div class="factor-bar"><div class="fill {bmi_cls}" style="width: {bmi_pct}%"></div></div>
      </div>
      <div class="factor-row">
        <div class="factor-top"><span class="fname">Tekanan darah (MAP {map_v:.1f})</span><span class="flevel">{bp_lvl}</span></div>
        <div class="factor-bar"><div class="fill {bp_cls}" style="width: {bp_pct}%"></div></div>
      </div>
      <div class="factor-row">
        <div class="factor-top"><span class="fname">Pola istirahat & tidur</span><span class="flevel">{tidur_lvl}</span></div>
        <div class="factor-bar"><div class="fill {tidur_cls}" style="width: {tidur_pct}%"></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Rekomendasi Medis
    st.markdown("""
    <div class="reco-card">
      <h3>Yang bisa Anda lakukan</h3>
      <div class="reco-item">
        <div class="reco-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3E8E7E" stroke-width="2">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
          </svg>
        </div>
        <div class="reco-text">
          <b>Periksa kadar HbA1C & Glukosa secara berkala</b>
          <span>Pemeriksaan laboratorium berkala membantu memantau perubahan kadar gula darah sejak dini.</span>
        </div>
      </div>
      <div class="reco-item">
        <div class="reco-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3E8E7E" stroke-width="2">
            <circle cx="12" cy="12" r="9"/>
            <path d="M12 7v5l3 3"/>
          </svg>
        </div>
        <div class="reco-text">
          <b>Jaga berat badan & aktivitas fisik rutin</b>
          <span>Latihan aerobik sedang (jalan cepat 30 menit/hari) efektif meningkatkan sensitivitas insulin.</span>
        </div>
      </div>
      <div class="reco-item">
        <div class="reco-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3E8E7E" stroke-width="2">
            <path d="M12 3a6 6 0 00-6 6c0 4 3 5 3 9h6c0-4 3-5 3-9a6 6 0 00-6-6z"/>
          </svg>
        </div>
        <div class="reco-text">
          <b>Perhatikan pola makan sehat & gizi seimbang</b>
          <span>Batasi konsumsi karbohidrat olahan dan perbanyak asupan serat dari sayur-sayuran hijau.</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 5. Tombol Aksi (CTA)
    if st.button("Cek ulang dengan data lain", type="secondary", use_container_width=True):
        go_to("landing")

    # 6. Accordion Metodologi Penilaian
    st.markdown("""
    <details class="methodology">
      <summary>Lihat metodologi penilaian ↓</summary>
      <div class="methodology-body">
        Hasil ini dihasilkan melalui dua tahap analisis Machine Learning: <b>Random Forest Tuned</b> untuk skrining awal yang peka mengidentifikasi potensi risiko (Recall tinggi), dan <b>XGBoost Tuned</b> untuk memperkuat ketepatan diagnosis (F1-score 0.7855 & ROC-AUC 0.956). Model dilatih menggunakan dataset kesehatan populasi NHANES berskala besar. Hasil ini tidak menggantikan pemeriksaan langsung oleh dokter.
      </div>
    </details>
    """, unsafe_allow_html=True)
