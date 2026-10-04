"""
Breast Cancer Classification Web Application
Powered by TensorFlow/Keras Multi-Layer Perceptron (MLP) & Streamlit
Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Classification - MLP",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-benign {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-malignant {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Feature Metadata & Statistics
# -----------------------------------------------------------------------------
FEATURE_INFO = [
    # Mean features
    {"id": "radius_mean", "name": "Radius (Mean)", "cat": "mean", "min": 6.0, "max": 30.0, "default": 14.12, "step": 0.01, "unit": "μm", "desc": "Mean distance from center to points on perimeter"},
    {"id": "texture_mean", "name": "Texture (Mean)", "cat": "mean", "min": 9.0, "max": 40.0, "default": 19.29, "step": 0.01, "unit": "", "desc": "Standard deviation of gray-scale values"},
    {"id": "perimeter_mean", "name": "Perimeter (Mean)", "cat": "mean", "min": 40.0, "max": 190.0, "default": 91.97, "step": 0.1, "unit": "μm", "desc": "Mean size of the core tumor perimeter"},
    {"id": "area_mean", "name": "Area (Mean)", "cat": "mean", "min": 140.0, "max": 2550.0, "default": 654.89, "step": 1.0, "unit": "μm²", "desc": "Mean area of the cell nuclei"},
    {"id": "smoothness_mean", "name": "Smoothness (Mean)", "cat": "mean", "min": 0.05, "max": 0.17, "default": 0.0964, "step": 0.0001, "unit": "", "desc": "Mean local variation in radius lengths"},
    {"id": "compactness_mean", "name": "Compactness (Mean)", "cat": "mean", "min": 0.01, "max": 0.35, "default": 0.1043, "step": 0.001, "unit": "", "desc": "Mean perimeter^2 / area - 1.0"},
    {"id": "concavity_mean", "name": "Concavity (Mean)", "cat": "mean", "min": 0.0, "max": 0.45, "default": 0.0888, "step": 0.001, "unit": "", "desc": "Mean severity of concave portions of contour"},
    {"id": "concave_points_mean", "name": "Concave Points (Mean)", "cat": "mean", "min": 0.0, "max": 0.21, "default": 0.0489, "step": 0.001, "unit": "", "desc": "Mean number of concave portions of contour"},
    {"id": "symmetry_mean", "name": "Symmetry (Mean)", "cat": "mean", "min": 0.1, "max": 0.31, "default": 0.1812, "step": 0.001, "unit": "", "desc": "Mean symmetry of cell nuclei"},
    {"id": "fractal_dimension_mean", "name": "Fractal Dimension (Mean)", "cat": "mean", "min": 0.04, "max": 0.10, "default": 0.0628, "step": 0.0001, "unit": "", "desc": "Mean 'coastline approximation' - 1"},

    # Standard Error (SE) features
    {"id": "radius_se", "name": "Radius (SE)", "cat": "se", "min": 0.1, "max": 3.0, "default": 0.405, "step": 0.001, "unit": "μm", "desc": "Standard error for the distance from center to perimeter"},
    {"id": "texture_se", "name": "Texture (SE)", "cat": "se", "min": 0.3, "max": 5.0, "default": 1.217, "step": 0.001, "unit": "", "desc": "Standard error for gray-scale standard deviation"},
    {"id": "perimeter_se", "name": "Perimeter (SE)", "cat": "se", "min": 0.7, "max": 22.0, "default": 2.866, "step": 0.01, "unit": "μm", "desc": "Standard error for cell perimeter"},
    {"id": "area_se", "name": "Area (SE)", "cat": "se", "min": 6.0, "max": 550.0, "default": 40.34, "step": 0.1, "unit": "μm²", "desc": "Standard error for cell area"},
    {"id": "smoothness_se", "name": "Smoothness (SE)", "cat": "se", "min": 0.001, "max": 0.035, "default": 0.0070, "step": 0.0001, "unit": "", "desc": "Standard error for radius variation"},
    {"id": "compactness_se", "name": "Compactness (SE)", "cat": "se", "min": 0.002, "max": 0.14, "default": 0.0255, "step": 0.0001, "unit": "", "desc": "Standard error for compactness"},
    {"id": "concavity_se", "name": "Concavity (SE)", "cat": "se", "min": 0.0, "max": 0.40, "default": 0.0319, "step": 0.0001, "unit": "", "desc": "Standard error for concavity"},
    {"id": "concave_points_se", "name": "Concave Points (SE)", "cat": "se", "min": 0.0, "max": 0.055, "default": 0.0118, "step": 0.0001, "unit": "", "desc": "Standard error for concave points"},
    {"id": "symmetry_se", "name": "Symmetry (SE)", "cat": "se", "min": 0.007, "max": 0.08, "default": 0.0205, "step": 0.0001, "unit": "", "desc": "Standard error for symmetry"},
    {"id": "fractal_dimension_se", "name": "Fractal Dimension (SE)", "cat": "se", "min": 0.0005, "max": 0.03, "default": 0.0038, "step": 0.0001, "unit": "", "desc": "Standard error for fractal dimension"},

    # Worst / Largest features
    {"id": "radius_worst", "name": "Radius (Worst)", "cat": "worst", "min": 7.0, "max": 37.0, "default": 16.27, "step": 0.01, "unit": "μm", "desc": "Largest/worst mean distance from center to perimeter"},
    {"id": "texture_worst", "name": "Texture (Worst)", "cat": "worst", "min": 12.0, "max": 50.0, "default": 25.68, "step": 0.01, "unit": "", "desc": "Worst standard deviation of gray-scale values"},
    {"id": "perimeter_worst", "name": "Perimeter (Worst)", "cat": "worst", "min": 50.0, "max": 255.0, "default": 107.26, "step": 0.1, "unit": "μm", "desc": "Worst cell nucleus perimeter"},
    {"id": "area_worst", "name": "Area (Worst)", "cat": "worst", "min": 180.0, "max": 4300.0, "default": 880.58, "step": 1.0, "unit": "μm²", "desc": "Worst cell nucleus area"},
    {"id": "smoothness_worst", "name": "Smoothness (Worst)", "cat": "worst", "min": 0.07, "max": 0.23, "default": 0.1324, "step": 0.0001, "unit": "", "desc": "Worst local variation in radius lengths"},
    {"id": "compactness_worst", "name": "Compactness (Worst)", "cat": "worst", "min": 0.02, "max": 1.1, "default": 0.2543, "step": 0.001, "unit": "", "desc": "Worst compactness"},
    {"id": "concavity_worst", "name": "Concavity (Worst)", "cat": "worst", "min": 0.0, "max": 1.3, "default": 0.2722, "step": 0.001, "unit": "", "desc": "Worst concavity"},
    {"id": "concave_points_worst", "name": "Concave Points (Worst)", "cat": "worst", "min": 0.0, "max": 0.30, "default": 0.1146, "step": 0.001, "unit": "", "desc": "Worst number of concave points"},
    {"id": "symmetry_worst", "name": "Symmetry (Worst)", "cat": "worst", "min": 0.15, "max": 0.67, "default": 0.2901, "step": 0.001, "unit": "", "desc": "Worst symmetry"},
    {"id": "fractal_dimension_worst", "name": "Fractal Dimension (Worst)", "cat": "worst", "min": 0.05, "max": 0.21, "default": 0.0839, "step": 0.0001, "unit": "", "desc": "Worst fractal dimension"}
]

FEATURE_NAMES = [f["id"] for f in FEATURE_INFO]

# Sample cases from actual dataset for quick testing
SAMPLE_BENIGN = {
    "radius_mean": 13.54, "texture_mean": 14.36, "perimeter_mean": 87.46, "area_mean": 566.3,
    "smoothness_mean": 0.09779, "compactness_mean": 0.08129, "concavity_mean": 0.06664,
    "concave_points_mean": 0.04781, "symmetry_mean": 0.1885, "fractal_dimension_mean": 0.05766,
    "radius_se": 0.2699, "texture_se": 0.7886, "perimeter_se": 2.058, "area_se": 23.56,
    "smoothness_se": 0.008462, "compactness_se": 0.0146, "concavity_se": 0.02387,
    "concave_points_se": 0.01315, "symmetry_se": 0.0198, "fractal_dimension_se": 0.0023,
    "radius_worst": 15.11, "texture_worst": 19.26, "perimeter_worst": 99.7, "area_worst": 711.2,
    "smoothness_worst": 0.144, "compactness_worst": 0.1773, "concavity_worst": 0.239,
    "concave_points_worst": 0.1288, "symmetry_worst": 0.2977, "fractal_dimension_worst": 0.07259
}

SAMPLE_MALIGNANT = {
    "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8, "area_mean": 1001.0,
    "smoothness_mean": 0.1184, "compactness_mean": 0.2776, "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471, "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095, "texture_se": 0.9053, "perimeter_se": 8.589, "area_se": 153.4,
    "smoothness_se": 0.006399, "compactness_se": 0.04904, "concavity_se": 0.05373,
    "concave_points_se": 0.01587, "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.6, "area_worst": 2019.0,
    "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119,
    "concave_points_worst": 0.2654, "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.1189
}

DATASET_MEANS = {f["id"]: f["default"] for f in FEATURE_INFO}


# -----------------------------------------------------------------------------
# 3. Model & Scaler Loader (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained neural network & scaler...")
def load_model_and_scaler():
    """Load pre-trained Keras model and StandardScaler."""
    model_path = "breast_cancer_model.keras"
    scaler_path = "scaler.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found in project directory.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file '{scaler_path}' not found in project directory.")

    scaler = joblib.load(scaler_path)

    try:
        import keras
        model = keras.models.load_model(model_path)
    except Exception:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)

    return model, scaler


# -----------------------------------------------------------------------------
# 4. Helper Inference Functions
# -----------------------------------------------------------------------------
def predict_single(input_dict, model, scaler):
    """
    Given a dictionary of 30 feature values, scale them and run model inference.
    Returns:
        prob_malignant: float (0.0 to 1.0)
        diagnosis: str ('Benign' or 'Malignant')
        confidence: float (50.0 to 100.0)
    """
    row = [input_dict[f_id] for f_id in FEATURE_NAMES]
    arr = np.array(row, dtype=np.float32).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prob_m = float(model.predict(arr_scaled, verbose=0)[0][0])
    diagnosis = "Malignant" if prob_m >= 0.5 else "Benign"
    confidence = (prob_m if prob_m >= 0.5 else (1.0 - prob_m)) * 100.0
    return prob_m, diagnosis, confidence


def predict_batch(df_input, model, scaler):
    """Run batch prediction on DataFrame containing required feature columns."""
    missing = [c for c in FEATURE_NAMES if c not in df_input.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    X = df_input[FEATURE_NAMES].values
    X_scaled = scaler.transform(X)
    probs = model.predict(X_scaled, verbose=0).ravel()
    
    results = df_input.copy()
    results["Malignancy_Probability"] = np.round(probs, 4)
    results["Predicted_Diagnosis"] = np.where(probs >= 0.5, "Malignant", "Benign")
    results["Confidence_%"] = np.round(np.where(probs >= 0.5, probs, 1.0 - probs) * 100, 2)
    return results


# -----------------------------------------------------------------------------
# 5. Sidebar Navigation & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-heart.png", width=64)
    st.title("Breast Cancer AI")
    st.caption("Deep Learning Diagnostic Assistant")
    
    mode = st.radio(
        "Navigation",
        ["🔍 Single Prediction", "📁 Batch CSV Prediction", "ℹ️ Model & Dataset Info"],
        index=0
    )

    st.markdown("---")
    st.subheader("Model Specifications")
    st.markdown("""
    - **Architecture:** MLP (Dense 16 → 8 → 1)
    - **Activation:** ReLU, Sigmoid
    - **Scaler:** `StandardScaler` (30 features)
    - **Target:** Benign (0) / Malignant (1)
    - **Framework:** TensorFlow / Keras 3
    """)

    st.markdown("---")
    st.caption("💡 *Disclaimer: This system is built for research and educational purposes. It should not replace professional medical evaluation.*")


# -----------------------------------------------------------------------------
# 6. Main Content Area
# -----------------------------------------------------------------------------

# Load model and scaler
try:
    model, scaler = load_model_and_scaler()
    models_ready = True
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")
    models_ready = False


if models_ready:
    # -------------------------------------------------------------------------
    # MODE 1: Single Prediction
    # -------------------------------------------------------------------------
    if mode == "🔍 Single Prediction":
        st.markdown('<div class="main-header">🩺 Breast Cancer Diagnostic Classifier</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Enter the 30 FNA cell nucleus measurements or load a preset sample to classify whether a tumor is <b>Benign</b> or <b>Malignant</b>.</div>',
            unsafe_allow_html=True
        )

        # Preset selection buttons
        st.markdown("##### ⚡ Quick Presets")
        col_b1, col_b2, col_b3 = st.columns(3)
        
        with col_b1:
            if st.button("🟢 Load Benign Sample Case", use_container_width=True):
                for k, v in SAMPLE_BENIGN.items():
                    st.session_state[f"input_{k}"] = v
                st.rerun()

        with col_b2:
            if st.button("🔴 Load Malignant Sample Case", use_container_width=True):
                for k, v in SAMPLE_MALIGNANT.items():
                    st.session_state[f"input_{k}"] = v
                st.rerun()

        with col_b3:
            if st.button("⚪ Reset to Dataset Averages", use_container_width=True):
                for k, v in DATASET_MEANS.items():
                    st.session_state[f"input_{k}"] = v
                st.rerun()

        st.markdown("---")

        # Organize 30 features into 3 tabs
        tab_mean, tab_se, tab_worst = st.tabs([
            "📊 Mean Measurements (10)",
            "📈 Standard Error Measurements (10)",
            "🔬 Worst / Largest Measurements (10)"
        ])

        current_inputs = {}

        # 1. Mean Tab
        with tab_mean:
            st.caption("Average values of cell nucleus characteristics computed from digitized FNA images.")
            cols = st.columns(2)
            mean_feats = [f for f in FEATURE_INFO if f["cat"] == "mean"]
            for i, feat in enumerate(mean_feats):
                col_idx = i % 2
                val_key = f"input_{feat['id']}"
                if val_key not in st.session_state:
                    st.session_state[val_key] = feat["default"]

                with cols[col_idx]:
                    val = st.number_input(
                        label=f"{feat['name']} ({feat['unit']})" if feat['unit'] else feat['name'],
                        min_value=float(feat["min"]),
                        max_value=float(feat["max"]),
                        value=float(st.session_state[val_key]),
                        step=float(feat["step"]),
                        key=val_key,
                        help=feat["desc"]
                    )
                    current_inputs[feat["id"]] = val

        # 2. SE Tab
        with tab_se:
            st.caption("Standard error values representing measurement variability and contour irregularity.")
            cols = st.columns(2)
            se_feats = [f for f in FEATURE_INFO if f["cat"] == "se"]
            for i, feat in enumerate(se_feats):
                col_idx = i % 2
                val_key = f"input_{feat['id']}"
                if val_key not in st.session_state:
                    st.session_state[val_key] = feat["default"]

                with cols[col_idx]:
                    val = st.number_input(
                        label=f"{feat['name']} ({feat['unit']})" if feat['unit'] else feat['name'],
                        min_value=float(feat["min"]),
                        max_value=float(feat["max"]),
                        value=float(st.session_state[val_key]),
                        step=float(feat["step"]),
                        key=val_key,
                        help=feat["desc"]
                    )
                    current_inputs[feat["id"]] = val

        # 3. Worst Tab
        with tab_worst:
            st.caption("Mean of the three largest values for each feature, capturing extreme cell nucleus abnormalities.")
            cols = st.columns(2)
            worst_feats = [f for f in FEATURE_INFO if f["cat"] == "worst"]
            for i, feat in enumerate(worst_feats):
                col_idx = i % 2
                val_key = f"input_{feat['id']}"
                if val_key not in st.session_state:
                    st.session_state[val_key] = feat["default"]

                with cols[col_idx]:
                    val = st.number_input(
                        label=f"{feat['name']} ({feat['unit']})" if feat['unit'] else feat['name'],
                        min_value=float(feat["min"]),
                        max_value=float(feat["max"]),
                        value=float(st.session_state[val_key]),
                        step=float(feat["step"]),
                        key=val_key,
                        help=feat["desc"]
                    )
                    current_inputs[feat["id"]] = val

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Run Neural Network Diagnosis", type="primary", use_container_width=True)

        if predict_btn or "last_prediction" in st.session_state:
            if predict_btn:
                prob_malignant, diagnosis, confidence = predict_single(current_inputs, model, scaler)
                st.session_state["last_prediction"] = {
                    "prob_malignant": prob_malignant,
                    "diagnosis": diagnosis,
                    "confidence": confidence,
                    "inputs": current_inputs.copy()
                }
            else:
                cached = st.session_state["last_prediction"]
                prob_malignant = cached["prob_malignant"]
                diagnosis = cached["diagnosis"]
                confidence = cached["confidence"]

            st.markdown("---")
            st.subheader("📋 Diagnostic Results")

            # Layout: Left side diagnosis badge & gauge, Right side key metrics comparison
            res_left, res_right = st.columns([1, 1], gap="large")

            with res_left:
                if diagnosis == "Malignant":
                    st.error(f"### ⚠️ Result: **MALIGNANT** (Cancerous)")
                    st.markdown(
                        f"The neural network predicts this sample exhibits malignant cellular characteristics with a **{confidence:.1f}% confidence**."
                    )
                else:
                    st.success(f"### ✅ Result: **BENIGN** (Non-Cancerous)")
                    st.markdown(
                        f"The neural network predicts this sample is non-cancerous/benign with a **{confidence:.1f}% confidence**."
                    )

                # Gauge / Probability Meter
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_malignant * 100,
                    number={'suffix': "%", 'font': {'size': 36}},
                    title={'text': "Malignancy Probability", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                        'bar': {'color': "#EF4444" if prob_malignant >= 0.5 else "#10B981"},
                        'steps': [
                            {'range': [0, 30], 'color': "#DCFCE7"},
                            {'range': [30, 70], 'color': "#FEF3C7"},
                            {'range': [70, 100], 'color': "#FEE2E2"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

                col_m1, col_m2 = st.columns(2)
                col_m1.metric("P(Benign)", f"{(1.0 - prob_malignant) * 100:.2f}%")
                col_m2.metric("P(Malignant)", f"{prob_malignant * 100:.2f}%")

            with res_right:
                st.markdown("##### 🔬 Comparative Feature Analysis")
                st.caption("Comparison of key measurements against dataset averages for Benign vs Malignant tumors.")

                # Radar chart of normalized key features
                key_radar_features = [
                    ("radius_mean", "Radius"),
                    ("texture_mean", "Texture"),
                    ("perimeter_mean", "Perimeter"),
                    ("area_mean", "Area"),
                    ("concavity_mean", "Concavity"),
                    ("concave_points_mean", "Concave Pts")
                ]

                # Benchmark averages (from WDBC dataset)
                benign_bench = [12.15, 17.91, 78.08, 462.8, 0.046, 0.026]
                malignant_bench = [17.46, 21.60, 115.37, 978.4, 0.161, 0.088]

                categories = [label for _, label in key_radar_features]
                patient_vals = []
                for feat_id, _ in key_radar_features:
                    patient_vals.append(current_inputs.get(feat_id, 0.0))

                # Normalize to 0-100 scale using min and max of each feature
                def norm(vals, feat_list):
                    res = []
                    for v, (fid, _) in zip(vals, feat_list):
                        info = next(f for f in FEATURE_INFO if f["id"] == fid)
                        n = (v - info["min"]) / (info["max"] - info["min"]) * 100.0
                        res.append(max(0.0, min(100.0, n)))
                    return res

                fig_radar = go.Figure()

                fig_radar.add_trace(go.Scatterpolar(
                    r=norm(benign_bench, key_radar_features),
                    theta=categories,
                    fill='toself',
                    name='Benign Baseline',
                    line_color='#10B981',
                    opacity=0.3
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=norm(malignant_bench, key_radar_features),
                    theta=categories,
                    fill='toself',
                    name='Malignant Baseline',
                    line_color='#EF4444',
                    opacity=0.3
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=norm(patient_vals, key_radar_features),
                    theta=categories,
                    fill='toself',
                    name='Current Sample',
                    line_color='#3B82F6',
                    line_width=3
                ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)
                    ),
                    showlegend=True,
                    height=280,
                    margin=dict(l=40, r=40, t=20, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

    # -------------------------------------------------------------------------
    # MODE 2: Batch CSV Prediction
    # -------------------------------------------------------------------------
    elif mode == "📁 Batch CSV Prediction":
        st.markdown('<div class="main-header">📁 Batch Breast Cancer Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Upload a CSV file containing patient sample features to generate high-throughput automated classifications.</div>',
            unsafe_allow_html=True
        )

        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        with col_u2:
            st.markdown("##### 📥 Sample Data")
            if os.path.exists("sample_test_data.csv"):
                with open("sample_test_data.csv", "rb") as f:
                    st.download_button(
                        label="Download Sample Test CSV",
                        data=f,
                        file_name="sample_test_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            st.caption("Includes 10 real patient samples (5 Benign, 5 Malignant) for demonstration.")

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.write(f"**Loaded dataset:** `{input_df.shape[0]}` rows and `{input_df.shape[1]}` columns.")

                # Check if all required features are present
                missing_cols = [c for c in FEATURE_NAMES if c not in input_df.columns]
                if missing_cols:
                    st.error(f"❌ Missing required feature columns in CSV ({len(missing_cols)}): {missing_cols[:5]}...")
                else:
                    with st.spinner("Processing batch predictions with Neural Network..."):
                        results_df = predict_batch(input_df, model, scaler)

                    st.success("✅ Batch predictions successfully completed!")

                    # Summary Metrics
                    total_samples = len(results_df)
                    benign_count = (results_df["Predicted_Diagnosis"] == "Benign").sum()
                    malignant_count = (results_df["Predicted_Diagnosis"] == "Malignant").sum()
                    malignancy_rate = (malignant_count / total_samples) * 100

                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Total Patients", total_samples)
                    m_col2.metric("Benign Diagnoses", benign_count, delta="Safe")
                    m_col3.metric("Malignant Diagnoses", malignant_count, delta="-Risk", delta_color="inverse")
                    m_col4.metric("Malignancy Rate", f"{malignancy_rate:.1f}%")

                    # Pie chart
                    fig_pie = px.pie(
                        names=["Benign", "Malignant"],
                        values=[benign_count, malignant_count],
                        color=["Benign", "Malignant"],
                        color_discrete_map={"Benign": "#10B981", "Malignant": "#EF4444"},
                        title="Diagnosis Distribution",
                        hole=0.4
                    )
                    fig_pie.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Display predictions table
                    st.markdown("##### 📄 Prediction Table")
                    display_cols = ["Predicted_Diagnosis", "Confidence_%", "Malignancy_Probability"] + [c for c in input_df.columns if c not in ["Predicted_Diagnosis", "Confidence_%", "Malignancy_Probability"]]
                    st.dataframe(results_df[display_cols], use_container_width=True)

                    # Download results CSV
                    csv_export = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Predictions as CSV",
                        data=csv_export,
                        file_name="breast_cancer_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )

            except Exception as e:
                st.error(f"Error processing uploaded CSV: {e}")

    # -------------------------------------------------------------------------
    # MODE 3: Model & Dataset Info
    # -------------------------------------------------------------------------
    elif mode == "ℹ️ Model & Dataset Info":
        st.markdown('<div class="main-header">ℹ️ Model Architecture & Dataset Documentation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Comprehensive technical background on the Multi-Layer Perceptron (MLP) and Wisconsin Breast Cancer Dataset.</div>',
            unsafe_allow_html=True
        )

        col_info1, col_info2 = st.columns(2)

        with col_info1:
            st.markdown("### 🧠 Neural Network Architecture")
            st.markdown("""
            The classification engine uses a feed-forward Multilayer Perceptron (MLP) built with TensorFlow/Keras:

            - **Input Layer:** 30 continuous numerical features (standardized via `StandardScaler`)
            - **Hidden Layer 1:** 16 neurons with **ReLU** activation
            - **Hidden Layer 2:** 8 neurons with **ReLU** activation
            - **Output Layer:** 1 neuron with **Sigmoid** activation (outputs $P(\text{Malignant})$)

            **Training Parameters:**
            - **Optimizer:** Adam
            - **Loss Function:** Binary Cross-Entropy
            - **Epochs:** 50
            - **Batch Size:** 16
            - **Validation Split:** 20%
            - **Test Set Accuracy:** ~98.2%
            """)

        with col_info2:
            st.markdown("### 📊 Dataset Overview (WDBC)")
            st.markdown("""
            The **Breast Cancer Wisconsin (Diagnostic)** dataset was created by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian:

            - **Total Instances:** 569
            - **Class Distribution:**
              - 357 Benign (62.7%)
              - 212 Malignant (37.3%)
            - **Features:** 30 real-valued inputs computed from digitized fine needle aspirate (FNA) images of breast masses.
            - **Categories:**
              1. *Mean:* Average values of cell nuclei characteristics
              2. *Standard Error (SE):* Standard error of cell nuclei characteristics
              3. *Worst:* Largest/worst values of cell nuclei characteristics
            """)

        st.markdown("---")
        st.markdown("### 📐 Feature Summary Reference")
        summary_table = [
            {"Measurement": "Radius", "Description": "Mean distances from center to perimeter points"},
            {"Measurement": "Texture", "Description": "Standard deviation of gray-scale pixel intensities"},
            {"Measurement": "Perimeter", "Description": "Total perimeter length of cell nuclei contours"},
            {"Measurement": "Area", "Description": "Surface area of cell nuclei"},
            {"Measurement": "Smoothness", "Description": "Local variation in radius lengths"},
            {"Measurement": "Compactness", "Description": "Calculated as perimeter² / area - 1.0"},
            {"Measurement": "Concavity", "Description": "Severity and depth of concave portions of contour"},
            {"Measurement": "Concave Points", "Description": "Number of contour concave indentations"},
            {"Measurement": "Symmetry", "Description": "Bilateral symmetry metric of cell contours"},
            {"Measurement": "Fractal Dimension", "Description": "Coastline approximation fractal analysis (-1)"},
        ]
        st.table(pd.DataFrame(summary_table))
