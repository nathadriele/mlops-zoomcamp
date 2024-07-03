import os
import pandas as pd
from datetime import datetime
from io_utils import get_input_path, get_output_path, save_data

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)

def create_test_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),     
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    return pd.DataFrame(data, columns=columns)

def run_integration_test():
    # Configurar variáveis de ambiente
    os.environ['INPUT_FILE_PATTERN'] = 's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'
    os.environ['OUTPUT_FILE_PATTERN'] = 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'
    os.environ['S3_ENDPOINT_URL'] = 'http://localhost:4566'

    year, month = 2022, 1
    input_file = get_input_path(year, month)
    
    # Criar e salvar dados de teste
    df_input = create_test_data()
    save_data(df_input, input_file)

    # Executar o script batch.py
    os.system(f'python main.py {year} {month}')

    # Ler e verificar resultados
    output_file = get_output_path(year, month)
    df_output = pd.read_parquet(output_file, storage_options={'client_kwargs': {'endpoint_url': os.environ['S3_ENDPOINT_URL']}})

    duration_sum = df_output['predicted_duration'].sum()
    print(f'Soma das durações previstas: {duration_sum}')

if __name__ == '__main__':
    run_integration_test()