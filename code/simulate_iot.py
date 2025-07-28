import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

def load_real_data(file_path='data/punjab_crop_data.csv'):
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}. Place your CSV in /data.")
            return None
        df = pd.read_csv(file_path)
        required_columns = ['Temperature (°C)', 'Humidity (%)', 'Rainfall (mm)', 'Yield (tonnes/ha)']
        if not all(col in df.columns for col in required_columns):
            print("Warning: CSV missing required columns. Adjust names or data.")
            return None
        print(f"Loaded {len(df)} rows from {file_path}.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def run_simulation(data, num_readings=10):
    if data is None:
        return None
    simulated_data = data.head(num_readings).copy()
    for i in range(len(simulated_data)):
        print(f"Simulated IoT Reading {i+1}: {simulated_data.iloc[i].to_dict()}")
        time.sleep(1)
    return simulated_data

def ai_predict_yield(data):
    if data is None or len(data) < 2:
        print("Error: Insufficient data for AI training.")
        return None
    X = data[['Temperature (°C)', 'Humidity (%)', 'Rainfall (mm)']]
    y = data['Yield (tonnes/ha)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"AI Model Trained. Mean Squared Error: {mse:.2f} (lower is better).")
    new_input = pd.DataFrame({'Temperature (°C)': [32.0], 'Humidity (%)': [85], 'Rainfall (mm)': [50]})
    predicted_yield = model.predict(new_input)[0]
    print(f"Predicted Yield for new weather (Temp 32°C, Humidity 85%, Rainfall 50mm): {predicted_yield:.2f} tonnes/ha")
    return predictions, y_test

if __name__ == "__main__":
    data = load_real_data()
    simulated = run_simulation(data, num_readings=50)
    if simulated is not None:
        ai_predict_yield(data)
    if not os.path.exists('results'):
        os.makedirs('results')
    data.to_csv('results/processed_data.csv', index=False)
    print("Results saved to results/processed_data.csv.")
