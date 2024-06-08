import requests
import pandas as pd
from io import BytesIO
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from typing import Dict
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verifica se os decoradores estão disponíveis
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_loader
def ingest_files(*args, **kwargs) -> pd.DataFrame:
    """Ingest data from a URL and return a pandas dataframe"""
    url = 'https://github.com/nathadriele/datasets/blob/main/yellow_tripdata_2023-03.parquet'
    response = requests.get(url)
    
    try:
        response.raise_for_status()
        logger.info("Data fetched successfully from %s", url)
    except requests.exceptions.RequestException as e:
        logger.error("Failed to fetch data from %s. %s", url, e)
        raise Exception(f'Failed to fetch data from {url}. {e}')
    
    df = pd.read_parquet(BytesIO(response.content))
    return df

@transformer
def read_dataframe(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """Read and preprocess the dataframe"""
    try:
        df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
        df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

        df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical_cols = ['PULocationID', 'DOLocationID']
        df[categorical_cols] = df[categorical_cols].astype(str)
        
        logger.info("Dataframe preprocessed successfully")
    except Exception as e:
        logger.error("Error in preprocessing the dataframe: %s", e)
        raise
    
    return df

@transformer
def train_model(df: pd.DataFrame, *args, **kwargs) -> Dict[str, object]:
    """Train a linear regression model on the preprocessed data"""
    try:
        df = read_dataframe(df)
        
        # Create features
        dict_list = df[['PULocationID', 'DOLocationID']].apply(lambda row: row.to_dict(), axis=1).tolist()
        
        # Fit DictVectorizer
        vectorizer = DictVectorizer()
        X = vectorizer.fit_transform(dict_list)
        y = df['duration']
        
        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X, y)
        
        logger.info("Model trained successfully with intercept: %f", model.intercept_)
    except Exception as e:
        logger.error("Error in training the model: %s", e)
        raise
    
    return {'vectorizer': vectorizer, 'model': model}

@data_exporter
def register_model(model_artifacts: Dict[str, object], *args, **kwargs) -> None:
    """Register the model and artifacts with MLflow"""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("export5_yellow_taxi_regression")
    mlflow.sklearn.autolog()
    
    try:
        with mlflow.start_run():
            mlflow.log_metric("intercept", model_artifacts['model'].intercept_)
            mlflow.sklearn.log_model(model_artifacts['model'], "model")
            mlflow.sklearn.log_model(model_artifacts['vectorizer'], "vectorizer")
            logger.info("Model and vectorizer registered with MLflow successfully")
    except Exception as e:
        logger.error("Error in registering the model with MLflow: %s", e)
        raise