from data_processing import read_data
from model_utils import load_model, predict
from io_utils import get_input_path, get_output_path, save_data

def main(year, month, categorical):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    dv, lr = load_model('model.bin')

    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    y_pred = predict(lr, dv, df[categorical])

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(df_result, output_file)

if __name__ == '__main__':
    year = 2024
    month = 3
    categorical = ['PULocationID', 'DOLocationID']
    main(year, month, categorical)