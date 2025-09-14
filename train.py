import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hides TensorFlow warnings/info messages

import tensorflow as tf
import matplotlib.pyplot as plt
from sys import exit as sys_exit
from warnings import catch_warnings

from Include.config_loader import load_config
from Include.errors import loader_error_msg_formatter
from Include.pipeline import load_normalize_split
from Include.nn_class import TFNeuralNetworkClassifier


def main():

    n_threads = os.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(n_threads)

    # Create output dir
    output_dir = "Model_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load configuration
    config_path = "config.yaml"
    try:
        with catch_warnings(record=True) as w_list:
            cfg = load_config(config_path)
            for w in w_list:
                print(w.message)
    except Exception as e:
        print(loader_error_msg_formatter(e, config_path))
        return 1

    #Load, normalize, and split the input data if they do not already exist
    df_train, _ = load_normalize_split(cfg.input_file_path, cfg.test_size, cfg.seed)

    models_metrics=[]
    model_names=[]
    for i, parameters in enumerate(cfg.model_parameters):

        model_name = parameters.name if (parameters.name and isinstance(parameters.name, str) and parameters.name.strip()) else str(i+1)
        model_names.append(model_name)
        print(f"\n\033[92m### Training model: {model_name} ###\033[0m")

        model_dir = os.path.join(output_dir, f"{model_name}")
        os.makedirs(model_dir, exist_ok=True)
        training_metrics_dir = os.path.join(model_dir, f"{model_name}_traing_metrics")
        os.makedirs(training_metrics_dir, exist_ok=True)

        nn_classifier = TFNeuralNetworkClassifier(
            epochs=parameters.epochs,
            batch_size=parameters.batch_size,
            learning_rate=parameters.learning_rate,
            early_stop_patience=parameters.early_stop_patience,
            lr_plateau_reduction_patience=parameters.lr_plateau_reduction_patience,
            conv_layers=parameters.conv_layers,
            conv_pool_size=parameters.conv_pool_size,
            hidden_layers=parameters.hidden_layers,
            dropout_rate=parameters.dropout_layers
        )

        metrics = nn_classifier.train(df_train.loc[:, parameters.use_features], df_train.loc[:, 0])
        models_metrics.append(metrics)
        nn_classifier.save_model(os.path.join(model_dir, f'{model_name}.keras'))

        plt.rcParams['figure.dpi'] = 500
        nn_classifier.plot_loss().savefig(os.path.join(training_metrics_dir, f'{model_name}_loss.png'))
        nn_classifier.plot_accuracy().savefig(os.path.join(training_metrics_dir, f'{model_name}_accuracy.png'))
        nn_classifier.plot_validation_roc().savefig(os.path.join(training_metrics_dir, f'{model_name}_validation_roc.png'))
    
    for name, metrics in zip(model_names, models_metrics):
        print(f"Model {name}:")
        print(f"\t Validation loss: {metrics[0]}")
        print(f"\t Validation accuracy: {metrics[1]}")
        print(f"\t Validation roc auc: {metrics[2]}")
    
    return 0


if __name__ == "__main__":
    sys_exit(main())