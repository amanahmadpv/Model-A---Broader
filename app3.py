import streamlit as st
import joblib
import numpy as np

# Load model, features, and encoding map
model = joblib.load("random_forest_onehot_model.pkl")
features = joblib.load("features_model_a.pkl")  # Should contain the 6 feature names
fpno_mapping = joblib.load("fpno_map.pkl")

# Global mean for unknown FP No.
global_mean = np.mean(list(fpno_mapping.values()))

st.set_page_config(page_title="Welding Rod Predictor – Lean Model A", layout="centered")
st.title("🔩 Welding Rod Consumption Predictor")
st.markdown("**Model A**: Predict at order time using top 6 features. Input values below to get the estimated welding rod consumption in kg/ton.")

# --- Input Section ---

# FP No. selectbox
fp_no = st.selectbox("📄 FP No.", options=sorted(fpno_mapping.keys()))
encoded_fp = fpno_mapping.get(fp_no, global_mean)

# Casting weight
cast_weight = st.number_input("⚖️ Total Despatched Casting Weight (Ton)", min_value=0.0, step=0.1)

# Grade Type (Stainless = 1, else 0)
grade_stainless = st.checkbox("🔧 Is the Grade Type *Stainless*?", value=False)
grade_type_val = 1 if grade_stainless else 0

# RT Required (0 or 1)
rt_required = st.radio("🔍 RT Required?", options=[0, 1], horizontal=True)

# Sample Options
smp_bulk = st.checkbox("🧪 Is this a *Bulk* sample?", value=False)
smp_grade = st.checkbox("🧪 Is this a *Grade* sample?", value=False)

# --- Prediction ---
if st.button("🎯 Predict Welding Rod Consumption"):
    # Create input in feature order
    input_values = {
        'new_fp_no_encoded': encoded_fp,
        'TotDespCastWt(Ton)': cast_weight,
        'Grade_Type_Stainless': grade_type_val,
        'RT Req': rt_required,
        'Smp_Grade Sample': int(smp_grade),
        'Smp_Bulk': int(smp_bulk)
    }

    # Convert to array in same order as training
    X_input = np.array([input_values[feat] for feat in features]).reshape(1, -1)

    # Predict and inverse log
    pred_log = model.predict(X_input)[0]
    pred_actual = np.expm1(pred_log)

    st.success(f"📦 **Predicted Welding Rod Usage**: `{pred_actual:.2f} kg/ton`")

    st.markdown("---")
    st.caption("Developed by Aman Ahmad P V — MBA (Data Analytics), Pondicherry University")

