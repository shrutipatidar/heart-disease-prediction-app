'''import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from datetime import datetime
import os

# Load model, scaler, polynomial features transformer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('poly.pkl', 'rb') as poly_file:
    poly = pickle.load(poly_file)

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Risk Prediction")

# Sidebar Inputs
st.sidebar.header("Patient Input Features")
if st.sidebar.button("Reset Inputs"):
    st.experimental_rerun()

# Medical History & Lifestyle Inputs
def user_input_features():
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x else "Female")
    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3],
                              format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
    trestbps = st.sidebar.slider("Resting BP (mm Hg)", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [1, 0], format_func=lambda x: "Yes" if x else "No")
    restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
    thalach = st.sidebar.slider("Max Heart Rate", 70, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x else "No")
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 10.0, 1.0)
    slope = st.sidebar.selectbox("Slope", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.sidebar.selectbox("Major Vessels Colored", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: ["Normal", "Fixed", "Reversible"][x])

    # Medical History & Lifestyle
    smoking = st.sidebar.selectbox("Do you smoke?", ["No", "Yes"])
    alcohol = st.sidebar.selectbox("Alcohol consumption", ["None", "Occasional", "Frequent"])
    exercise_level = st.sidebar.selectbox("Exercise Level", ["Low", "Moderate", "High"])
    diet_quality = st.sidebar.selectbox("Diet Quality", ["Poor", "Average", "Good"])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal,
        'smoking': smoking, 'alcohol': alcohol, 'exercise_level': exercise_level, 'diet_quality': diet_quality
    }
    return data

input_data = user_input_features()
input_df = pd.DataFrame([input_data])

# Transform input for model
model_features = input_df.drop(['smoking', 'alcohol', 'exercise_level', 'diet_quality'], axis=1)
input_poly = poly.transform(model_features)
input_scaled = scaler.transform(input_poly)

# Prediction
probability = model.predict_proba(input_scaled)[0][1]
prediction = model.predict(input_scaled)[0]

# Save logs
log_entry = input_data.copy()
log_entry['prediction'] = prediction
log_entry['probability'] = round(probability, 2)
log_entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

log_file = "user_logs.csv"
if os.path.exists(log_file):
    existing_logs = pd.read_csv(log_file)
    existing_logs = pd.concat([existing_logs, pd.DataFrame([log_entry])], ignore_index=True)
else:
    existing_logs = pd.DataFrame([log_entry])

existing_logs.to_csv(log_file, index=False)

# Feature names for SHAP
poly_feature_names = poly.get_feature_names_out(model_features.columns)

# Result Display
col1, col2 = st.columns([2, 3])
with col1:
    st.subheader("Prediction Result")
    st.write(f"### Probability of Heart Disease: {probability * 100:.2f}%")
    if prediction == 1:
        st.error("⚠️ High Risk Detected")
        st.markdown("**🔎 Recommendation:** Please consult a cardiologist and consider lifestyle changes.")
    else:
        st.success("✅ Low Risk Detected")
        st.markdown("**💡 Tip:** Maintain a healthy routine, regular checkups, and balanced diet.")

    st.markdown("#### Personalized Lifestyle Suggestions")
    st.write(f"- Smoking: {input_data['smoking']}")
    st.write(f"- Alcohol: {input_data['alcohol']}")
    st.write(f"- Exercise: {input_data['exercise_level']}")
    st.write(f"- Diet: {input_data['diet_quality']}")

    if input_data['smoking'] == 'Yes':
        st.warning("🚭 Consider quitting smoking to reduce cardiovascular risk.")
    if input_data['exercise_level'] == 'Low':
        st.info("🏃‍♂️ Try to include at least 30 mins of daily moderate exercise.")
    if input_data['diet_quality'] == 'Poor':
        st.info("🥗 Consider shifting to a heart-healthy diet.")

    # Links
    st.markdown("#### 📞 Emergency Resources")
    st.markdown("- [🩺 Find a Cardiologist](https://www.apollospectra.com/specialities/cardiology)")
    st.markdown("- [🚑 Emergency Services (India)](tel:102)")

with col2:
    st.subheader("SHAP Model Explanation")
    background = scaler.transform(poly.transform(np.zeros((1, model_features.shape[1]))))
    explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")
    shap_values = explainer.shap_values(input_scaled)

    shap_exp = shap.Explanation(values=shap_values,
                                 base_values=explainer.expected_value,
                                 data=input_scaled,
                                 feature_names=poly_feature_names)

    # SHAP Bar Chart
    fig, ax = plt.subplots()
    shap.plots.bar(shap_exp, show=False)
    st.pyplot(fig)

    # SHAP Force Plot as iframe
    force_html = shap.force_plot(explainer.expected_value, shap_values[0], input_scaled[0],
                                 feature_names=poly_feature_names, matplotlib=False)
    components.html(force_html.html(), height=300)

with st.expander("📄 Show Raw Input Data"):
    st.write(input_df)

# Footer
st.markdown("---")
st.markdown("Developed by **Your Name** | ❤️ Heart Disease Predictor")

'''