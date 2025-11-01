# car-price-api-deploy

UsedCarValuator: Model Training Notebook

This notebook is the core of our project. We will go through the full data science lifecycle:

Load Data

Clean Data (This is the hardest part!)

Exploratory Data Analysis (EDA)

Feature Engineering

Preprocessing & Pipeline Building (The professional way)

Model Training & Comparison

Final Model Evaluation & Saving

Step 1: Get Your Data

Before you start, you need data.

Go to Kaggle and search for "Used Car Dataset India".

A great, popular choice is the "CarDekho Used Car Data" (you can find it here or here).

Download the CSV file (e.g., car data.csv or CAR DETAILS FROM CAR DEKHO.csv).

Place this file in your data/raw/ directory. For this guide, I'll assume the file is named cardekho_data.csv.

Step 2: Setup & Load Data

Let's import our libraries and load the dataset.

# --- Core Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import joblib # For saving the model

# --- Sklearn ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# --- Models ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Set visualization style
sns.set(style="whitegrid")

# Load the dataset
DATA_PATH = "../data/raw/cardekho_data.csv" # Update this to your file name
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}")
    print("Please download the dataset from Kaggle and place it in the data/raw/ directory.")
    # In a real notebook, you'd stop execution here


Step 3: Exploratory Data Analysis (EDA) & Cleaning

Let's understand our data. This is where we play detective.

# --- 3.1 Initial Inspection ---
print("--- First 5 Rows ---")
print(df.head())

print("\n--- Data Info ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())


Observations from .info():

selling_price is an int. This is good! Self-Correction: Many datasets (like the one I based the app.py on) have this as an object (e.g., "4.5 Lakhs"). This dataset is much cleaner. We'll adapt. If your selling_price is an object, you'll need to write a cleaning function for it.

Car_Name is an object. We'll need to extract the brand from this.

fuel, seller_type, transmission, owner are categorical.

There are no missing values in this specific dataset. This is rare and lucky. If you had them, you'd use df.fillna() or SimpleImputer in your pipeline.

Dataset-Specific Cleaning (if your price column is text):
If your selling_price was "4.5 Lakh", you would do this (we'll skip this as our data is clean):*

# # --- EXAMPLE CLEANING FUNCTION (if price is text) ---
# def clean_price(price):
#     if isinstance(price, str):
#         price = price.replace('₹', '').replace(',', '')
#         if 'Lakh' in price:
#             price = float(price.replace('Lakh', '')) * 100000
#         else:
#             price = float(price)
#     return int(price)
# 
# # df['selling_price'] = df['selling_price'].apply(clean_price)


Step 4: Feature Engineering

This is where we create new, more powerful features from the ones we have.

# --- 4.1 Create 'car_age' ---
# This is a very strong predictor.
current_year = datetime.date.today().year
df['car_age'] = current_year - df['year']

# --- 4.2 Extract 'brand' from 'Car_Name' ---
# 'Maruti Swift Dzire' -> 'Maruti'
df['brand'] = df['Car_Name'].apply(lambda x: x.split(' ')[0])

# --- 4.3 Clean 'owner' column ---
# Let's check the unique values
print("\n--- Owner unique values ---")
print(df['owner'].unique())
# We see 'Test Drive Car'. Let's group it with 'First Owner' for simplicity
df['owner'] = df['owner'].replace('Test Drive Car', 'First Owner')

# --- 4.4 Define our final features ---
# We can drop the original 'Car_Name' and 'year'
df_final = df.drop(columns=['Car_Name', 'year'])
print("\n--- Final DataFrame Head ---")
print(df_final.head())


Step 5: Visualize Data (Optional but Recommended)

Let's check our target variable, selling_price.

# --- 5.1 Visualize Price Distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(df_final['selling_price'], kde=True, bins=50)
plt.title('Distribution of Selling Price')
plt.show()


Observation: The plot is heavily right-skewed.  This is bad for linear models.

Pro-Tip: We'll apply a log transform to the price to make it more "normal" (bell-shaped). This will be our new target variable. We must remember to convert it back (using np.expm1()) when we evaluate.

# --- 5.2 Apply Log Transform ---
df_final['log_price'] = np.log1p(df_final['selling_price'])

# --- 5.3 Visualize Log-Transformed Price ---
plt.figure(figsize=(10, 6))
sns.histplot(df_final['log_price'], kde=True, bins=50)
plt.title('Distribution of Log-Transformed Selling Price')
plt.show()


Observation: Much better! This looks more like a normal distribution, which is ideal.

Step 6: Preprocessing & Building the Pipeline

This is the most professional part. We will build a ColumnTransformer that automatically applies different preprocessing steps to different columns. This prevents "data leakage" and makes our model ready for production.

# --- 6.1 Define Features (X) and Target (y) ---
# Our target is the 'log_price'
TARGET = 'log_price'
y = df_final[TARGET]

# Our features are all columns *except* the price and log_price
X = df_final.drop(columns=[TARGET, 'selling_price'])
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# --- 6.2 Split the Data ---
# We split *before* preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# --- 6.3 Identify Column Types ---
# We need to tell our pipeline which columns are numeric and which are categorical
numerical_cols = ['present_price', 'kms_driven', 'car_age']
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']


# --- 6.4 Create the Preprocessing Pipelines ---

# Pipeline for numerical features:
# 1. StandardScaler: Scales data (e.g., puts 'kms_driven' and 'car_age' on the same scale)
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Pipeline for categorical features:
# 1. OneHotEncoder: Converts categories (e.g., 'Petrol', 'Diesel') into numbers (0s and 1s)
#    handle_unknown='ignore' tells it to ignore categories it hasn't seen in training
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


# --- 6.5 Build the Master Preprocessor ---
# This ColumnTransformer applies the right transformer to the right columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ],
    remainder='passthrough' # Keep any columns not listed (though we've listed them all)
)


Step 7: Model Training & Comparison

Now we'll create full pipelines that include our preprocessor and our model.

# --- 7.1 Model 1: Linear Regression (Baseline) ---
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

pipeline_lr.fit(X_train, y_train)


# --- 7.2 Model 2: Random Forest ---
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

pipeline_rf.fit(X_train, y_train)


# --- 7.3 Model 3: XGBoost (The Powerhouse) ---
pipeline_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

pipeline_xgb.fit(X_train, y_train)


Step 8: Model Evaluation

Let's see which model performed best on our unseen X_test data.

# --- 8.1 Create an Evaluation Function ---
def evaluate_model(model, X_test, y_test):
    # Predict log_price
    y_pred_log = model.predict(X_test)
    
    # --- CRITICAL ---
    # Convert log_price predictions and actuals back to original scale (Rupees)
    y_pred_actual = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)
    
    # Calculate R-squared (higher is better)
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    # Calculate Root Mean Squared Error (lower is better)
    # This is the average error of our prediction in Rupees
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    
    return r2, rmse

# --- 8.2 Evaluate All Models ---
r2_lr, rmse_lr = evaluate_model(pipeline_lr, X_test, y_test)
r2_rf, rmse_rf = evaluate_model(pipeline_rf, X_test, y_test)
r2_xgb, rmse_xgb = evaluate_model(pipeline_xgb, X_test, y_test)

# --- 8.3 Display Results ---
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'R-Squared': [r2_lr, r2_rf, r2_xgb],
    'RMSE (Rupees)': [rmse_lr, rmse_rf, rmse_xgb]
})

print("\n--- Model Comparison ---")
print(results.sort_values(by='R-Squared', ascending=False))


Expected Output:
You will likely see that XGBoost and Random Forest have the highest R-Squared (e.g., > 0.95) and the lowest RMSE. XGBoost is often the winner.

Step 9: Save the Final Model

We've chosen our winner (let's assume it's XGBoost). Now we save the entire pipeline to a single file. This file contains the preprocessor and the trained model, which is exactly what our app.py needs.

# --- 9.1 Define the Winning Pipeline ---
# (You can also retrain on the FULL dataset for a final boost, but this is fine)
final_model_pipeline = pipeline_xgb 

# --- 9.2 Define the Save Path ---
MODEL_SAVE_PATH = "../src/model.pkl" 

# --- 9.3 Save the Model ---
try:
    joblib.dump(final_model_pipeline, MODEL_SAVE_PATH)
    print(f"\nModel successfully saved to {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"Error saving model: {e}")

# --- 9.4 (Optional) Test Loading the Model ---
try:
    loaded_model = joblib.load(MODEL_SAVE_PATH)
    print("\nModel loaded successfully for a test.")
    
    # Test with a sample
    sample = X_test.iloc[0:1]
    prediction = loaded_model.predict(sample)
    actual_price = np.expm1(prediction[0])
    print(f"Test prediction on one sample: ₹{actual_price:,.0f}")
    
except Exception as e:
    print(f"Error loading saved model: {e}")



Congratulations!

You now have a file named model.pkl in your src/ folder. This is your complete, trained, and production-ready model pipeline.

Your next step is to update app/app.py to use this model, but the skeleton I gave you is already set up to look for this exact file. You should be able to just run python app/app.py and it will work! (You may need to tweak the predict_price function in app.py to match the exact columns: present_price, kms_driven, car_age, fuel, etc.)
