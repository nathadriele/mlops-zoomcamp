import pickle
import pandas as pd
import numpy as np
import sys

def read_data(filename):
    df = pd.read_parquet(filename)
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].fillna(-1).astype('int').astype('str')
    return df

def predict_duration(model, dv, df):
    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <year> <month>")
        sys.exit(1)

    year = int(sys.argv[1])
    month = int(sys.argv[2])

    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    df = read_data(filename)
    y_pred = predict_duration(model, dv, df)

    std_dev = np.std(y_pred)
    print(f'The standard deviation of the predicted duration is: {std_dev:.2f}')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predictions': y_pred})

    output_file = f'output_file_{year:04d}-{month:02d}.parquet'
    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)
    print(f'DataFrame saved as Parquet: {output_file}')

    print(f'The expected average duration is: {y_pred.mean():.2f}')

if __name__ == '__main__':
    main()
    
#python homework_deployment.py 2023 4
#python homework_deployment.py 2023 5