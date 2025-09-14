import pandas as pd
import warnings
import os

from tensorflow import keras

from Include.errors import loader_error_msg_formatter
from Include.config_loader import load_config
from Include.nn_class import TFNeuralNetworkClassifier


def test_loader(config_path, output_dir):
    try:
        with warnings.catch_warnings(record=True):
            cfg = load_config(config_path)
    except Exception as e:
        raise RuntimeError(loader_error_msg_formatter(e, config_path))
        
    try:
        df_test = pd.read_csv("test.csv", header=None)
    except Exception:
        raise RuntimeError("Impossible to read the 'test.csv'")

    if not os.path.exists(output_dir):
        raise RuntimeError(f"Impossible to read the trained models, {output_dir} don't found")
        
    model_par=[]
    for par in cfg.model_parameters:
        model_par.append({
            "name": par.name,
            "path": os.path.join(output_dir, f"{par.name}", f"{par.name}.keras"),
            "batch_size": par.batch_size,
            "epochs": par.epochs,
            "use_features": par.use_features
        })
    return df_test, model_par


def model_loader(path, epochs):
    try:
        with warnings.catch_warnings(record=True) as wlist:
            model = TFNeuralNetworkClassifier(model=keras.models.load_model(path))
            for w in wlist:
                if not (
                isinstance(w.message, UserWarning)
                and "Skipping variable loading for optimizer" in str(w.message) and  epochs==0):
                    warnings.warn(w.message, w.category)    
    except Exception as e:
        raise RuntimeError(f"Error loading model {path}: {e}")
    
    return model


def make_prediction(df_test, use_feature, model, model_name, prediction_dir):
    X_test = df_test.loc[:, use_feature]
    y_test = df_test.loc[:, 0]

    y_pred_prob = model.predict_prob(X_test)
    y_pred_class = (y_pred_prob > 0.5).astype(int)

    result = {"y_test": y_test.values,
              "y_pred_prob": y_pred_prob,
              "y_pred_class": y_pred_class}
    
    os.makedirs(prediction_dir, exist_ok=True)
    prediction_df = pd.DataFrame({
        "y_test": y_test.tolist(),
        "Y_pred_prob": y_pred_prob.flatten().tolist(),
        "y_pred_class": y_pred_class.flatten().tolist()
    })

    prediction_df.to_csv(os.path.join(prediction_dir, f"{model_name}_predictions.csv"), index=False)

    return result
