🚗 Used Car Price Predictor
A full-stack machine learning application that predicts the resale value of used cars in India based on various technical and historical parameters. This project combines a robust XGBoost regression pipeline with a responsive React frontend.

👥 The Team
[salmon6934](https://github.com/salmon6934): Frontend Development (React, UI/UX, API Integration
[Nishita1652](https://github.com/Nishita1652): Machine Learning Engineering (Data Cleaning, EDA, Model Training & Pipeline)

🚀 Features
Real-time Prediction: Get instant valuation for used cars.
Advanced ML Pipeline: Uses log-transformation and Scikit-Learn pipelines to ensure high accuracy ($R^2 > 0.95$).
Responsive UI: Modern interface designed to capture car specifications like mileage, fuel type, and brand.
Robust Preprocessing: Automatically handles categorical encoding and feature scaling.

🛠️ Tech Stack
Frontend: React.js, CSS3
Backend/API: Flask / FastAPI (Python)
Machine Learning: Python, Scikit-Learn, XGBoost, Pandas, Numpy
DevOps: Joblib (Model Serialization)

📂 Project Structure
├── data/
│   └── raw/               # Raw CarDekho datasets
├── src/
│   ├── model.pkl          # Trained & serialized model pipeline
│   └── train_model.ipynb  # Core training notebook
├── app/
│   └── app.py             # Backend API script
└── frontend/              # React source code

🧠 Machine Learning Workflow
The model follows a rigorous data science lifecycle:
Data Cleaning: Handled missing values and standardized price formats.
Feature Engineering: Created high-impact features like car_age and extracted brand from car names.
Log Transformation: Applied$$y_{log} = \ln(1 + y)$$to the target variable to handle right-skewed price distribution.
Pipeline Building: Integrated StandardScaler and OneHotEncoder into a ColumnTransformer to prevent data leakage.
Model Selection: Evaluated Linear Regression, Random Forest, and XGBoost (Winner).

📊 Evaluation Results
The final XGBoost model achieved the following performance on the test set:
R-Squared: ~0.97
RMSE: Lower than baseline models, indicating high precision in price estimation.
