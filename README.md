# 🩺 Diabetes Risk Screening Web App

> **AI-powered diabetes risk assessment tool** combining clinical parameters, medical history, and lifestyle factors through a dual-model architecture (Random Forest & XGBoost) trained on NHANES dataset.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-00758F?style=flat)](https://xgboost.ai)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20Now-4CAF50?style=flat)](https://predict-your-diabet.streamlit.app/)

---

## 📖 Overview

This project transforms my undergraduate thesis research on diabetes prediction into an **interactive web application** for public health screening. It implements a **dual-model inference pipeline** that balances sensitivity (early detection via Random Forest) and diagnostic precision (XGBoost confirmation), delivering transparent, actionable risk assessments to users in under 2 minutes.

**Key Features:**
- **Multi-step guided wizard** for demographics, clinical measurements, mental health, and lifestyle data
- **Dual-model architecture**: RF (high recall for screening) + XGBoost (balanced F1/ROC-AUC for diagnosis)
- **Visual risk breakdown**: Dynamic factor bars showing HbA1c, BMI, blood pressure, and sleep pattern contributions
- **Clinical transparency**: Methodology disclosure with model performance metrics (F1: 0.7855, ROC-AUC: 0.956)
- **Privacy-first design**: No data storage, instant client-side processing

---

## 🎯 Problem & Motivation

Diabetes affects 463 million adults globally (IDF, 2019), with many cases undiagnosed until complications arise. Traditional risk calculators often:
- Rely on limited features (age, BMI, family history only)
- Lack model transparency for clinical validation
- Present results in non-intuitive formats for laypeople

This app addresses these gaps by:
1. Incorporating **20 evidence-based risk factors** (metabolic, psychological, behavioral)
2. Using **ensemble models** to reduce false negatives while maintaining diagnostic accuracy
3. Providing **educational result screens** that explain risk drivers and next steps

---

## 🏗️ Architecture

### System Design
```
User Input (Web Form)
    ↓
Preprocessing Pipeline (utils/preprocessing.py)
  • Binary encoding (gender, medical history)
  • One-hot encoding (race/ethnicity, drop_first=True)
  • Alcohol frequency remapping (NHANES codes 0-10)
  • Mean Arterial Pressure calculation
    ↓
Feature Scaling (MinMaxScaler, fitted on training data)
    ↓
Dual-Model Inference
  ├─ Random Forest (clf_rf.pkl) → Screening Mode (High Recall)
  └─ XGBoost (clf_xgb.pkl) → Diagnostic Mode (Balanced F1 & ROC-AUC)
    ↓
Risk Categorization Logic
  • High Risk: XGB ≥50% OR RF ≥65%
  • Moderate Risk: RF ≥35% OR XGB ≥25%
  • Low Risk: Below moderate thresholds
    ↓
Result Visualization (Streamlit UI)
```

### Model Performance (Validation Set)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest (Tuned)** | 0.8124 | 0.7201 | 0.8891 | 0.7961 | 0.957 |
| **XGBoost (Tuned)** | 0.8219 | 0.7512 | 0.8231 | **0.7855** | **0.956** |

*Training dataset: NHANES 2015-2018 (n=5,635 after preprocessing)*

### Tech Stack
- **Frontend**: Streamlit (multi-page wizard, responsive layout)
- **ML Framework**: scikit-learn (Random Forest), XGBoost
- **Preprocessing**: pandas, NumPy
- **Styling**: Custom CSS (Source Serif 4 + Inter typefaces, health-tech color palette)
- **Deployment**: Streamlit Community Cloud (auto-deploy from GitHub)

---

## 📊 Feature Engineering Pipeline

The model requires **20 input features** processed from user-friendly form fields:

| Category | Features | Transformations |
|----------|----------|-----------------|
| **Demographics** | `gender`, `usia`, `ras` (race/ethnicity) | Binary encoding (gender), one-hot encoding (race, 5 dummies) |
| **Clinical Measurements** | `BMI`, `HbA1c`, `kadar_kolesterol`, `tekanan_darah` (MAP) | MAP = diastolic + (systolic - diastolic)/3 |
| **Medical History** | `riw_kolesterol_tinggi`, `riw_liver`, `riw_tiroid`, `riw_kanker` | Binary flags (0/1) |
| **Mental Health** | `sedih-depresi-putus_asa`, `gangguan_tidur`, `gangguan_makan` | Ordinal scales (0-3) |
| **Lifestyle** | `alkohol`, `merokok100` | NHANES code remapping, binary smoking flag |

**Critical Implementation Detail**: The `scaler.pkl` artifact is fitted on the original training set (`X_train`) and **must not be re-fitted** on new input data to maintain distribution consistency.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Virtual environment tool (venv/conda)

### Installation

1. **Clone repository**
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction-web-app.git
   cd diabetes-prediction-web-app
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run application**
   ```bash
   streamlit run app.py
   ```

5. **Access web app**
   - Local: `http://localhost:8501`
   - Network: `http://<your-ip>:8501`

### Project Structure
```
diabetes-prediction-web-app/
├── app.py                    # Main Streamlit application
├── style.css                 # Custom CSS styling
├── requirements.txt          # Python dependencies
├── model/
│   ├── clf_rf.pkl           # Random Forest (Screening Mode)
│   ├── clf_xgb.pkl          # XGBoost (Diagnostic Mode)
│   └── scaler.pkl           # MinMaxScaler (fitted on training data)
├── utils/
│   └── preprocessing.py     # Feature engineering & transformation
└── .streamlit/
    └── config.toml          # Streamlit configuration
```

---

## 🎨 Design Philosophy

### Visual Design Principles
- **Accessibility**: 18px base font, high contrast ratios, Source Serif 4 (headings) + Inter (body) for readability
- **Progressive Disclosure**: Multi-step wizard to prevent cognitive overload during form completion
- **Transparent AI**: Dual-model comparison and factor contribution bars demystify black-box predictions

### UX Flow
1. **Landing Screen**: Value proposition + trust signals (privacy guarantee, NHANES dataset badge)
2. **Form Wizard (4 Steps)**: Demographics → Clinical → Mental Health → Lifestyle
3. **Result Screen**: Risk badge → Dual-model probabilities → Factor breakdown → Clinical recommendations

---

## 🧪 Testing & Validation

### Quick Test Scenarios
The app includes preset profiles accessible from the landing screen's "Quick Test" section:

**Low Risk Profile**
- Female, 28 years, Asian ethnicity
- HbA1c: 5.0%, BMI: 21.8, BP: 115/75
- No medical history, healthy lifestyle
- **Expected**: RF ~15%, XGB ~8%, Low Risk badge

**High Risk Profile**
- Male, 56 years, Hispanic ethnicity
- HbA1c: 7.4%, BMI: 32.5, BP: 145/92
- History of high cholesterol + thyroid disorder
- Frequent depression, daily sleep disturbance, active smoker
- **Expected**: RF ~78%, XGB ~62%, High Risk badge

### Model Verification
```python
# Run preprocessing validation
python -m utils.preprocessing

# Expected output: 20-feature DataFrame with correct column order
# ['gender', 'usia', 'BMI', 'HbA1c', 'kadar_kolesterol', 'alkohol',
#  'riw_kolesterol_tinggi', 'sedih_depresi', 'gangguan_tidur',
#  'gangguan_makan', 'riw_liver', 'riw_tiroid', 'riw_kanker', 'merokok100',
#  'tekanan_darah', 'ras_2', 'ras_3', 'ras_4', 'ras_6', 'ras_7']
```

---

## 🔬 Research Background

This application is built upon my undergraduate thesis research:

**Title**: *Comparison of K-Nearest Neighbors, XGBoost, Support Vector Machine, and Random Forest in Diabetes Prediction Based on Risk Factors*

**Dataset**: [National Health and Nutrition Examination Survey (NHANES) August 2021-August 2023](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023)
- Survey components: 11 datasets (Demographic, Blood Pressure, Body Measures, HbA1c, Cholesterol, Alcohol Use, Medical Conditions, Diabetes, Mental Health, Smoking)
- Original merged size: 11,933 respondents
- After preprocessing & filtering (binary classification, excluding prediabetes): 4,080 samples
- After SMOTE-ENN resampling: 4,117 training samples
- Features: 20 validated risk factors grouped into 5 categories (demographics, clinical measurements, medical history, mental health, lifestyle)

**Model Development**:
- **Comparison algorithms**: K-Nearest Neighbors, SVM, Random Forest, XGBoost
- **Optimization**: GridSearchCV with 10-Fold Stratified Cross-Validation
- **Class imbalance handling**: SMOTE-ENN (Synthetic Minority Oversampling + Edited Nearest Neighbors)

**Key Findings**:
1. XGBoost achieved best overall balance (F1: 0.7855, ROC-AUC: 0.956)
2. Random Forest excelled at recall (87.73%) for screening scenarios
3. Feature importance analysis (via ELI5) identified HbA1c, Age, and BMI as dominant risk factors

---

## ⚠️ Important Disclaimers

### Medical Disclaimer
This tool is designed for **educational and screening purposes only**. It is **not a substitute for professional medical diagnosis**. Key limitations:

- **Not FDA-approved**: This is a research prototype, not a certified medical device
- **Population bias**: Model trained primarily on U.S. NHANES data (generalization to other populations not validated)
- **Self-reported data**: Accuracy depends on user's knowledge of their own health metrics
- **Snapshot assessment**: Does not account for temporal changes or genetic predisposition not captured in features

**Always consult a qualified healthcare provider** for definitive diagnosis and treatment planning.

### Privacy & Data Handling
- **No data storage**: All computations run client-side; form inputs are not logged or transmitted
- **No cookies**: Application does not track user behavior or collect analytics
- **Session-based**: All data cleared on browser refresh or navigation away from app

---

## 🛣️ Future Enhancements

### Future Enhancements
- **Multi-language support** (Indonesian, English)
- **PDF report generation** for clinical consultation
- **Temporal tracking** (allow users to save results locally and compare over time)
- **Feature importance visualization** (SHAP values for individual predictions)
- **Model retraining pipeline** (automated updates with newer NHANES releases)

### Technical Debt
- [ ] Add unit tests for preprocessing pipeline (`pytest`)
- [ ] Implement CI/CD with GitHub Actions (linting, type checking, deployment)
- [ ] Containerize with Docker for reproducible local deployments
- [ ] Add performance monitoring (inference latency, model drift detection)

---

## 📚 References & Resources

### Dataset
- Centers for Disease Control and Prevention (CDC). [NHANES August 2021-August 2023](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023)

### Modeling Techniques
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*. [Paper](https://arxiv.org/abs/1603.02754)
- Xu, Z., Shen, D., Nie, T., & Kou, Y. (2020). A Hybrid Sampling Algorithm Combining M-SMOTE and ENN Based On Random Forest For Medical Imbalanced Data. *Journal of Biomedical Informatics*, 107. [Paper](https://doi.org/10.1016/j.jbi.2020.103465)


### Clinical Guidelines
- American Diabetes Association. (2025). What Is the A1C Test *American Diabetes Association*. [Paper](https://diabetes.org/about-diabetes/a1c)
- Centers for Disease Control and Prevention. (2024b, Maret 19). Adult BMI Categories. *Centers for Disease Control and Prevention*. [Link](https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html)
- Centers for Disease Control and Prevention. (2024c, Mei 15). About Cholesterol. *Centers for Disease Control and Prevention*.  [Link](https://www.cdc.gov/cholesterol/about/index.html)
- Centers for Disease Control and Prevention. (2024d, Mei 15). Diabetes Risk Factors. *Centers for Disease Control and Prevention*. [Link](https://www.cdc.gov/diabetes/risk-factors/index.html)
- Magliano, D. J., & Boyko, E. J. (2021). IDF Diabetes Atlas 10th edition (10
ed.). *International Diabetes Federation*. [Link](www.diabetesatlas.org)
- Magliano, D. J., & Boyko, E. J. (2025). IDF Diabetes Atlas - 11th edition | 2025. *International Diabetes Federation*. [Link](https://diabetesatlas.org/)
- World Health Organization. (2021). *Guideline for the pharmacological
treatment of hypertension in adults*. World Health Organization. [Link](https://www.who.int/publications/i/item/9789240033986)


---

## 🤝 Contributing

While this is primarily a portfolio project, I welcome:
- **Bug reports** via GitHub Issues
- **Feature suggestions** aligned with clinical best practices
- **Documentation improvements** for clarity

For major changes, please open an issue first to discuss proposed modifications.

---

## 👤 Author

**Your Name**  
AI Engineering Portfolio | Data Science Enthusiast

- GitHub: [@hibartaufik](https://github.com/hibartaufik)
- LinkedIn: [Hibar Taufikurachman](https://linkedin.com/in/hibar-taufikurachman)
- Email: hibartaufikurachman@gmail.com

---

## 🙏 Acknowledgments

- **Academic advisors** who guided the original thesis research
- **CDC NHANES Program** for providing open-access public health data
- **Streamlit Team** for the intuitive web framework
- **scikit-learn & XGBoost communities** for robust ML libraries

---

<div align="center">
  <p><strong>⚡ Built with Streamlit | Powered by Ensemble ML | Designed for Public Health Impact</strong></p>
  <p><em>Last updated: September 2026</em></p>
</div>
