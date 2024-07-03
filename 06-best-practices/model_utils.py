import pickle

def load_model(model_path):
    with open(model_path, 'rb') as f_in:
        return pickle.load(f_in)

def predict(model, dv, X):
    dicts = X.to_dict(orient='records')
    X_val = dv.transform(dicts)
    return model.predict(X_val)