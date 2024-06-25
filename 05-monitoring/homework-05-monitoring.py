import requests
import pandas as pd
from tqdm import tqdm
from evidently.metrics import ColumnQuantileMetric, ColumnValueRangeMetric
from evidently.report import Report
import json
import webbrowser
from pathlib import Path

def download_file(url, save_path):
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get("Content-Length", 0))
    
    with open(save_path, "wb") as handle, tqdm(
        desc=f"Downloading {save_path.name}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in resp.iter_content(chunk_size=1024):
            size = handle.write(data)
            progress_bar.update(size)

def load_and_process_data(file_path):
    data = pd.read_parquet(file_path)
    data['lpep_pickup_datetime'] = pd.to_datetime(data['lpep_pickup_datetime'])
    return data[data['lpep_pickup_datetime'].dt.month == 3]

def calculate_statistics(data):
    fare_amount_mean = data['fare_amount'].mean()
    print(f"Mean of the 'fare_amount' column: {fare_amount_mean:.2f}")

    data['day'] = data['lpep_pickup_datetime'].dt.date
    daily_quantiles = data.groupby('day')['fare_amount'].quantile(0.5)
    max_daily_quantile = daily_quantiles.max()
    print(f"The maximum value of the 0.5 quantile in the 'fare_amount' column during March 2024 is: {max_daily_quantile:.2f}")

def generate_report(data):
    empty_data = pd.DataFrame({col: pd.Series(dtype=data[col].dtype) for col in data.columns})
    report = Report(metrics=[
        ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
        ColumnValueRangeMetric(column_name="fare_amount", left=0, right=100) 
    ])
    report.run(current_data=data, reference_data=empty_data)
    report.save_html('dashboard_report.html')

def save_dashboard_config(config, file_path):
    with open(file_path, 'w') as f:
        json.dump(config, f)
    print(f"Panel configuration file saved in: {file_path}")

def main():
    FILE_NAME = 'green_tripdata_2024-03.parquet'
    BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    DATA_DIR = Path('./data')
    
    DATA_DIR.mkdir(exist_ok=True)
    
    file_url = f"{BASE_URL}/{FILE_NAME}"
    save_path = DATA_DIR / FILE_NAME
    download_file(file_url, save_path)
    
    data = load_and_process_data(save_path)
    
    calculate_statistics(data)
    
    generate_report(data)
    
    webbrowser.open('dashboard_report.html')
    
    config_path = Path('C:/Users/admin/Downloads/dashboards/New dashboard-1719253590737')
    save_dashboard_config(dashboard_config, config_path)

if __name__ == "__main__":
    main()