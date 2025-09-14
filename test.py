import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hides TensorFlow warnings/info messages
import tensorflow as tf

from sys import exit as sys_exit
from sklearn.metrics import classification_report, confusion_matrix

from Include.test_functions import test_loader, model_loader, make_prediction
from Include.plotter import plot_confusion_matrix, plot_roc_curve


def main():
    config_path = "config.yaml"
    output_dir = "Model_outputs"
    print("\033[94m[NOTE]\033[0m: Please ensure that config.yaml is the same file used during training.")
    try:
        df_test, model_par = test_loader(config_path, output_dir)
    except Exception as e:
        print(e)
        return 1

    models = []
    for par in model_par:
        try:
            path = par["path"]
            model= model_loader(path, par["epochs"])
            models.append(model)
            print(f"Loaded model {path}")
        except Exception as e:
            print(e)
            return 1

    print("\n\033[94mCalculating predictions...\033[0m")
    data_predictions=[]
    for par, model in zip(model_par, models):
        model_name=par["name"]
        prediction_dir = os.path.join(output_dir, model_name, f"{model_name}_prediction_result")

        prediction = make_prediction(df_test, par["use_features"], model, model_name, prediction_dir)
        data_predictions.append(prediction)

    for par, prediction in zip(model_par, data_predictions):
        model_name = par['name']
        y_test = prediction["y_test"]
        y_pred_prob = prediction["y_pred_prob"]
        y_pred_class = prediction["y_pred_class"]

        prediction_dir = os.path.join(output_dir, model_name, f"{model_name}_prediction_result")

        print(f"\n\033[92m###  Model {model_name} metrics ###\033[0m")
        print(classification_report(y_test, y_pred_class))
        print(pd.DataFrame(confusion_matrix(y_test, y_pred_class), 
                        index=['Actual 0', 'Actual 1'], 
                        columns=['Predicted 0', 'Predicted 1']))
        
        fig = plot_confusion_matrix(confusion_matrix(y_test, y_pred_class), title=f"{model_name} confusion matrix")
        fig.savefig(os.path.join(prediction_dir, f"{model_name}_confusion_matrix.png"), dpi=300)
        
        fig = plot_roc_curve(y_test, y_pred_prob, title=f"{model_name} ROC curve")
        fig.savefig(os.path.join(prediction_dir, f"{model_name}_roc_curve.png"), dpi=300)


if __name__ == "__main__":
    sys_exit(main())

