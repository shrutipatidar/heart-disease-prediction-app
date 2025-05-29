## Heart Disease Risk Prediction App

A Streamlit-based web application that predicts the risk of heart disease using a machine learning model. 
The app allows users to input health parameters, see prediction results with confidence levels, and explore model explanations via SHAP visualizations.

Features

- Interactive UI- for inputting patient health data.
- Probability prediction of heart disease risk with confidence levels.
- Model explanation using SHAP bar charts and force plots.
- Batch prediction via CSV upload.
- Input validation and health tips based on prediction results.
- Downloadable input data and prediction results as CSV.
- Logging and monitoring of predictions.


## Installation

1. Clone the repository:

git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

2. Create and activate a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate      # On Linux/macOS
.\venv\Scripts\activate       # On Windows

3. Install required Python packages:

pip install -r requirements.txt
Place your pre-trained model files (model.pkl, scaler.pkl, poly.pkl) in the project directory.

4. Usage
Run the Streamlit app locally:

streamlit run dashboard.py

## Files Description
dashboard.py: Main Streamlit app file.

model.pkl: Serialized trained ML model (ensure to add your own).

scaler.pkl: Scaler used to normalize input features.

poly.pkl: Polynomial features transformer.

requirements.txt: Lists required Python packages.

README.md: This documentation file.

## How to Train Your Own Model
You can retrain the model using a larger and more diverse dataset. The training pipeline includes data preprocessing, feature scaling, polynomial feature expansion, and model fitting (e.g., Logistic Regression or other classifiers). Save the trained model, scaler, and polynomial features objects as .pkl files and place them in the project folder for the app to use.

## Logging & Monitoring
User inputs and prediction results are logged to prediction_logs.csv to allow analysis and improve model performance over time. Please ensure to respect user privacy and data protection laws when handling logs.

## Acknowledgments
1. SHAP for model explanation
2. Streamlit for rapid app development
3. Dataset source: https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci?resource=download
