import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components  # NEW for rendering SHAP force plot

# Load model, scaler, polynomial features transformer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('poly.pkl', 'rb') as poly_file:
    poly = pickle.load(poly_file)

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("❤️ Heart Disease Risk Prediction")

# Sidebar: User inputs with reset
st.sidebar.header("Input Features")
if st.sidebar.button("Reset Inputs"):
    st.experimental_rerun()

# Add explanation markdown for chest pain types in sidebar
st.sidebar.markdown("ℹ️ **Chest Pain Types Explained:**")
st.sidebar.markdown("""
- **Typical Angina:** Chest pain related to heart disease.
- **Atypical Angina:** Chest pain not typical of heart disease.
- **Non-anginal Pain:** Chest pain not related to heart disease.
- **Asymptomatic:** No chest pain.
""")

def user_input_features():
    age = st.sidebar.slider("Age", 20, 100, 50,
                            help="Patient's age in years. Age is an important risk factor for heart disease.")
    sex = st.sidebar.selectbox("Sex", options=[1, 0],
                               format_func=lambda x: "Male" if x == 1 else "Female",
                               help="1 = Male, 0 = Female")
    cp = st.sidebar.selectbox("Chest Pain Type",
                              options=[0, 1, 2, 3],
                              format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}[x],
                              help="Type of chest pain experienced by the patient")
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120,
                                help="Resting blood pressure (mm Hg). High values increase risk.")
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200,
                             help="Serum cholesterol in mg/dl. High cholesterol is a risk factor.")
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0],
                              format_func=lambda x: "Yes" if x == 1 else "No",
                              help="Is fasting blood sugar > 120 mg/dl?")
    restecg = st.sidebar.selectbox("Resting ECG Results", options=[0, 1, 2],
                                  format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x],
                                  help="Resting electrocardiographic results")
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 220, 150,
                               help="Maximum heart rate achieved during exercise")
    exang = st.sidebar.selectbox("Exercise Induced Angina", options=[1, 0],
                                format_func=lambda x: "Yes" if x == 1 else "No",
                                help="Exercise induced angina?")
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 10.0, 1.0,
                               help="ST depression induced by exercise relative to rest")
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                                format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x],
                                help="Slope of the peak exercise ST segment")
    ca = st.sidebar.selectbox("Number of Major Vessels Colored", options=[0, 1, 2, 3, 4],
                             help="Number of major vessels colored by fluoroscopy (0-4)")
    thal = st.sidebar.selectbox("Thalassemia", options=[1, 2, 3],
                               format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x],
                               help="Thalassemia type")

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return data

input_data = user_input_features()
input_df = pd.DataFrame([input_data])

# Input validation warnings
if input_data['chol'] > 400:
    st.sidebar.warning("⚠️ Serum cholesterol above 400 mg/dl is considered very high.")

if input_data['age'] < 30 and input_data['cp'] == 3:
    st.sidebar.warning("⚠️ Asymptomatic chest pain in younger patients is uncommon. Please double-check your input.")

# Prepare data and do prediction inside try-except for error handling
try:
    input_array = np.array(input_df.iloc[0])
    input_poly = poly.transform([input_array])
    input_scaled = scaler.transform(input_poly)

    # Prediction and probability
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    # Confidence message
    if probability > 0.75:
        confidence = "High confidence"
    elif 0.5 < probability <= 0.75:
        confidence = "Medium confidence"
    else:
        confidence = "Low confidence"

except Exception as e:
    st.error(f"❌ An error occurred during prediction: {e}")
    st.stop()

# Get feature names from polynomial transformer for SHAP
poly_feature_names = poly.get_feature_names_out(input_df.columns)

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Prediction Result")

    st.write(f"### Probability of Heart Disease: {probability*100:.2f}%")
    st.write(f"**Model Confidence:** {confidence}")

    if prediction == 1:
        st.error("⚠️ High risk of heart disease detected.")
        st.info("**Health Tip:** Please consult a healthcare professional for advice.")
    else:
        st.success("✅ Low risk of heart disease detected.")
        st.info("**Health Tip:** Keep up a healthy lifestyle and routine checkups.")

    # Probability gauge
    fig, ax = plt.subplots(figsize=(5, 1))
    ax.barh(['Risk Probability'], [probability], color='red' if probability > 0.5 else 'green')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title('Risk Probability Gauge')
    plt.tight_layout()
    st.pyplot(fig)

    # Download user input data button
    csv = input_df.to_csv(index=False).encode()
    st.download_button(label="Download Input Data as CSV", data=csv, file_name='input_data.csv', mime='text/csv')

with col2:
    st.subheader("Model Explanation (SHAP)")

    # Prepare background dataset for SHAP explainer (zero array)
    background = np.zeros((1, input_df.shape[1]))
    background_poly = poly.transform(background)
    background_scaled = scaler.transform(background_poly)

    # Create SHAP explainer
    explainer = shap.LinearExplainer(model, background_scaled, feature_perturbation="interventional")

    shap_values = explainer.shap_values(input_scaled)

    # Create SHAP Explanation object with correct poly feature names
    shap_exp = shap.Explanation(values=shap_values,
                               base_values=explainer.expected_value,
                               data=input_scaled,
                               feature_names=poly_feature_names)

    # --- SHAP bar chart ---
    ax = shap.plots.bar(shap_exp, show=False)
    fig = ax.get_figure()  # get parent figure from axes
    fig.tight_layout()
    st.pyplot(fig)

    # --- UPDATED: SHAP force plot for local explanation ---
    st.subheader("Local Explanation (Force Plot)")
    force_plot = shap.force_plot(
        explainer.expected_value, shap_values, input_scaled, feature_names=poly_feature_names, matplotlib=False
    )
    force_plot_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    components.html(force_plot_html, height=400, scrolling=True)

    # --- SHAP summary plot (global explanation) ---
    st.subheader("Global Explanation (Summary Plot)")
    try:
        shap.summary_plot(shap_values, input_scaled, feature_names=poly_feature_names, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.error(f"Error displaying summary plot: {e}")

with st.expander("Show Raw Input Data"):
    st.write(input_df)

# --- Batch Prediction Section ---
st.markdown("---")
st.header("Batch Prediction: Upload CSV for Multiple Patients")

uploaded_file = st.file_uploader("Upload CSV file with patient data", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:", batch_df.head())

        # Validate columns
        expected_cols = list(input_df.columns)
        if not all(col in batch_df.columns for col in expected_cols):
            st.error(f"CSV must contain these columns: {expected_cols}")
        else:
            # Prepare batch data
            batch_array = batch_df[expected_cols].values
            batch_poly = poly.transform(batch_array)
            batch_scaled = scaler.transform(batch_poly)

            # Predictions
            batch_probs = model.predict_proba(batch_scaled)[:, 1]
            batch_preds = model.predict(batch_scaled)

            batch_df['Heart Disease Probability'] = batch_probs
            batch_df['Prediction'] = batch_preds

            st.write("Prediction results:", batch_df)

            # SHAP explanations for batch
            batch_shap_values = explainer.shap_values(batch_scaled)

            st.subheader("SHAP Summary Plot for Uploaded Data")
            shap.summary_plot(batch_shap_values, batch_scaled, feature_names=poly_feature_names, show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf()

            # Download result with predictions
            csv_result = batch_df.to_csv(index=False).encode()
            st.download_button("Download Predictions as CSV", csv_result, "batch_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

st.markdown("---")
st.markdown("Developed by **Shruti Patidar** | Heart Disease Risk Prediction App")

