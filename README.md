# 🩺 Breast Cancer Classification using Deep Learning (MLP)

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

An end-to-end Deep Learning diagnostic assistant built with **TensorFlow / Keras** and **Streamlit**. The system classifies fine needle aspirate (FNA) breast mass cell nucleus samples into **Benign** or **Malignant** with high accuracy (~98.2%) based on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

---

## 🌟 Features

- **Interactive Diagnostic Web UI**: Modern, responsive Streamlit dashboard with intuitive feature input sliders and numeric controls.
- **Trained Neural Network & Scaler Integration**: Uses pre-trained `breast_cancer_model.keras` and `scaler.pkl` with cached loading (`@st.cache_resource`) for sub-second inference.
- **Organized Feature Layout**: 30 features neatly categorized across 3 tabs:
  - 📊 **Mean Measurements** (10 features)
  - 📈 **Standard Error (SE)** (10 features)
  - 🔬 **Worst / Extreme Measurements** (10 features)
- **⚡ Quick-Test Presets**: One-click preset buttons to load sample Benign or Malignant patient data or reset to dataset averages.
- **Interactive Risk Visualizations**:
  - Malignancy probability gauge with risk tiering (Low, Moderate, High Risk).
  - Plotly Radar chart comparing the current patient's profile against Benign and Malignant population baselines.
- **📁 Batch CSV Predictions**:
  - Upload CSV files with multiple patient records.
  - Generates instant batch classifications with confidence metrics and distribution charts.
  - One-click export to download results with predicted diagnoses.
- **Technical Documentation Tab**: In-app overview of model architecture, layer configurations, and feature metadata.

---

## 📂 Project Structure

```plaintext
breast-cancer-classification-mlp/
├── app.py                      # Interactive Streamlit Web Application
├── breast_cancer_model.keras   # Saved TensorFlow/Keras MLP neural network
├── scaler.pkl                  # Fitted StandardScaler for 30 features
├── main.ipynb                  # Training, EDA, and evaluation notebook (Google Colab)
├── sample_test_data.csv        # 10 sample patient cases (5 Benign, 5 Malignant) for testing
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
├── wdbc.data                   # Original Wisconsin Breast Cancer dataset
└── wdbc.names                  # Dataset attributes and descriptions
```

---

## 🧠 Model Architecture & Training

The Multi-Layer Perceptron (MLP) was trained using TensorFlow/Keras on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset:

```
Input (30 features)
       │
       ▼
[ Dense(16, activation='relu') ]
       │
       ▼
[ Dense(8, activation='relu') ]
       │
       ▼
[ Dense(1, activation='sigmoid') ] ──> P(Malignant) ∈ [0, 1]
```

### Hyperparameters & Settings:
- **Input Dimension:** 30 features (standardized via `StandardScaler`)
- **Hidden Layers:** Layer 1 (16 units, ReLU), Layer 2 (8 units, ReLU)
- **Output Layer:** 1 unit, Sigmoid ($0 = \text{Benign}, 1 = \text{Malignant}$)
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Batch Size:** 16
- **Epochs:** 50
- **Validation Split:** 20%
- **Evaluation Accuracy:** **~98.25%** on unseen test data

---

## 📊 Dataset Features (30 Attributes)

Each tumor sample is characterized by 10 core measurements computed for cell nuclei:

| Feature | Description |
| :--- | :--- |
| **Radius** | Mean of distances from center to points on perimeter |
| **Texture** | Standard deviation of gray-scale values |
| **Perimeter** | Total length of cell nucleus border |
| **Area** | Surface area enclosed by nucleus border |
| **Smoothness** | Local variation in radius lengths |
| **Compactness** | Computed as $(\text{perimeter}^2 / \text{area}) - 1.0$ |
| **Concavity** | Severity of concave portions of contour |
| **Concave Points** | Number of concave portions of contour |
| **Symmetry** | Bilateral symmetry measure of nucleus |
| **Fractal Dimension** | "Coastline approximation" - 1 |

For each core feature, three statistics are computed:
1. **Mean** (`*_mean`)
2. **Standard Error** (`*_se`)
3. **Worst / Extreme** (`*_worst` - average of 3 largest values)

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/breast-cancer-classification-mlp.git
cd breast-cancer-classification-mlp
```

### 3. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
The application will launch in your browser automatically at:
`http://localhost:8501`

---

## 🧪 Testing the Application

### Single Sample Diagnosis:
1. Open the **🔍 Single Prediction** view.
2. Click **🟢 Load Benign Sample Case** or **🔴 Load Malignant Sample Case** to immediately populate the input fields with verified patient records.
3. Click **🔮 Run Neural Network Diagnosis** to view the diagnosis, confidence score, probability gauge, and comparative radar chart.

### Batch CSV Diagnosis:
1. Navigate to **📁 Batch CSV Prediction** in the sidebar.
2. Click **Download Sample Test CSV** or upload your own CSV file containing the 30 WDBC features.
3. Review the automated diagnostic metrics, pie chart breakdown, and download the full annotated results CSV.

---

## ⚠️ Medical Disclaimer

This software is developed strictly for **educational, experimental, and research purposes**. It is not certified for clinical use or direct patient medical diagnosis. All diagnostic determinations must be made by qualified healthcare professionals and supported by clinical biopsy and pathological examinations.
