# 🫀 Heart Disease Predictor

A machine learning pipeline to predict heart disease risk using clinical data. Built for accuracy, interpretability, and offline usability.

## 📌 Project Overview

This project leverages structured health data to train predictive models that assess the likelihood of heart disease. It includes preprocessing, model training, evaluation, and deployment-ready assets.

- **Dataset**: [Heart Disease UCI](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **Models Used**: Logistic Regression, Random Forest
- **Accuracy**: Up to 88.6% with Random Forest
- **Deployment Assets**: Trained model (`.pkl`), scaler, and user input template

## 🧠 Features

- Handles missing values via mean imputation
- One-hot encoding for categorical variables
- Feature scaling with `StandardScaler`
- Model persistence using `joblib`
- Predicts on custom user data via CSV
- Visualizations for feature importance and confusion matrix

## 🛠️ Tech Stack

- Python 3.12
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Google Colab

## 📂 Project Structure

heart-disease/ 
- heart_disease_uci.csv # Original dataset 
- heart_rf_model.pkl # Trained Random Forest model 
- heart_scaler.pkl # Scaler used for preprocessing 
- Heart_user_template.csv # Sample input format for predictions 
- Heart_Disease_Predictor.ipynb # Main notebook


## 🚀 How to Run

1. Clone the repo and open the notebook in Google Colab.
2. Upload `heart_disease_uci.csv` or use the Kaggle API to download it.
3. Run all cells to train and evaluate models.
4. Upload a user dataset matching the template to generate predictions.

## 📊 Model Performance

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | 84.2%   | 88%       | 85%    | 87%      |
| Random Forest       | 88.6%   | 90%       | 87%    | 88%      |

## 📈 Visual Insights

- **Confusion Matrix**: Evaluates classification performance
- **Feature Importance**: Highlights top predictors (e.g., `thalch`, `oldpeak`, `ca`)

## 📦 Deployment Ready

- `heart_rf_model.pkl`: Load with `joblib` for inference
- `heart_scaler.pkl`: Ensures consistent preprocessing
- `Heart_user_template.csv`: Format for user input

## 🙌 Acknowledgments
Dataset by Redwankarimsony on Kaggle

## 🧪 Sample Prediction Code

```python
import joblib
import pandas as pd

# Load user data
user_df = pd.read_csv('your_user_data.csv')

# Preprocess and encode
# ... (match training features)

# Load model and scaler
model = joblib.load('heart_rf_model.pkl')
scaler = joblib.load('heart_scaler.pkl')

# Predict
user_df_scaled = scaler.transform(user_df)
preds = model.predict(user_df_scaled)
user_df['Heart_Disease_Prediction'] = preds




