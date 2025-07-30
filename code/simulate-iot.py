import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import boto3

def load_real_data(file_name='punjab_crop_data.csv'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))
    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}. Place your CSV in /data.")
        return None
    df = pd.read_csv(file_path)

    cols_to_clean = ['Yield (tonnes/ha)', 'Production (tonnes)', 'Area (hectares)']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    weather_cols = ['Temperature (°C)', 'Humidity (%)', 'Rainfall (mm)']
    for col in weather_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=cols_to_clean + weather_cols)

    required_columns = ['Temperature (°C)', 'Humidity (%)', 'Rainfall (mm)', 'Yield (tonnes/ha)']
    if not all(col in df.columns for col in required_columns):
        print("Warning: CSV missing required columns after cleaning. Adjust names or data.")
        return None

    print(f"Loaded {len(df)} rows from {file_path} after cleaning.")
    return df

def run_simulation(dataframe, num_readings=100):
    if dataframe is None:
        return None
    simulation_subset = dataframe.head(num_readings).copy()
    for idx in range(len(simulation_subset)):
        print(f"Simulated IoT Reading {idx + 1}: {simulation_subset.iloc[idx].to_dict()}")
        time.sleep(0.5)
    return simulation_subset

def ai_predict_yield(dataframe):
    if dataframe is None or len(dataframe) < 10:
        print("Error: Insufficient data for AI training (need at least 10 rows).")
        return None
    features = dataframe[['Temperature (°C)', 'Humidity (%)', 'Rainfall (mm)']]
    targets = dataframe['Yield (tonnes/ha)']
    features_train, features_test, targets_train, targets_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features_train, targets_train)
    predictions = model.predict(features_test)
    mse = mean_squared_error(targets_test, predictions)
    print(f"AI Model Trained (Random Forest). Mean Squared Error: {mse:.2f} (lower is better).")
    new_sample = pd.DataFrame({'Temperature (°C)': [32.0], 'Humidity (%)': [85], 'Rainfall (mm)': [50]})
    predicted_yield = model.predict(new_sample)[0]
    print(f"Predicted Yield for new weather (Temp 32°C, Humidity 85%, Rainfall 50mm): {predicted_yield:.2f} tonnes/ha")
    results_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    plot_path = os.path.join(results_folder, 'prediction_plot.png')
    plt.scatter(targets_test, predictions)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title('AI Prediction Accuracy')
    plt.savefig(plot_path)
    print(f"Prediction plot saved to {plot_path}.")
    return predictions, targets_test

def upload_to_aws(file_path=None, bucket_name=None, region=None):
    if file_path is None or bucket_name is None:
        print("AWS upload skipped: file_path or bucket_name not provided.")
        return
    try:
        if region:
            s3_client = boto3.client('s3', region_name=region)
        else:
            s3_client = boto3.client('s3')
        s3_client.upload_file(file_path, bucket_name, os.path.basename(file_path))
        print(f"Results uploaded to AWS S3 bucket: {bucket_name}.")
    except Exception as error:
        print(f"Error uploading to AWS: {error}. Check credentials and bucket name.")

if __name__ == "__main__":
    main_data = load_real_data()
    simulated_readings = run_simulation(main_data, num_readings=100)
    if simulated_readings is not None:
        ai_predict_yield(main_data)
    results_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    processed_csv_path = os.path.join(results_dir, 'processed_data.csv')
    if main_data is not None:
        main_data.to_csv(processed_csv_path, index=False)
        print(f"Results saved to {processed_csv_path}.")
        upload_to_aws(file_path=processed_csv_path, bucket_name='punjab-iot-ai-results-2025', region='eu-north-1')
