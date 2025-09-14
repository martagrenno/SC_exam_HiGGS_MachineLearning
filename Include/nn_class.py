import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.utils.validation import check_X_y, check_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np


class TFNeuralNetworkClassifier:
    """
    TensorFlow-based neural network classifier compatible with scikit-learn API.
    Supports flexible architecture via 'hidden_layers'.
    """

    def __init__(self, epochs=20, batch_size=32, learning_rate=0.001, early_stop_patience=5,
                 lr_plateau_reduction_patience=5, conv_layers=None, conv_pool_size=None, hidden_layers=(64,),
                 dropout_rate=(0.3,), model=None):
        """
        Initialize the classifier object.

        Parameters
        ----------
        epochs : int, optional (default=20)
            Number of training epochs.
        batch_size : int, optional (default=32)
            Batch size for training.
        learning_rate : float, optional (default=0.001)
            Optimizer learning rate.
        early_stop_patience : int, optional (default=5)
            Number of epochs for EarlyStopping patience.
        lr_plateau_reduction_patience : int, optional (default=5)
            Number of epochs for ReduceLROnPlateau patience.
        conv_layers : list or tuple, optional
            List/tuple of (filters, kernel_size) for each convolutional layer.
        conv_pool_size : list or tuple, optional
            List/tuple of MaxPooling size after every convolution layer.
        hidden_layers : tuple or list, optional (default=(64,))
            Number of neurons for each hidden layer.
        dropout_rate : tuple or list, optional (default=(0.3,))
            Dropout rate for each hidden layer.
        model : keras.Model, optional
            Pre-built Keras model to use.
        """
        self.type = type
        self.input_dim = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_plateau_reduction_patience = lr_plateau_reduction_patience
        self.early_stop_patience = early_stop_patience
        self.conv_layers = conv_layers # List/tuple of list/tuple of the kind (filter, kernel_size)
        self.conv_pool_size = conv_pool_size # List or tuple of the MaxPooling after every convolution layer
        self.hidden_layers = hidden_layers  # List or tuple of layer sizes
        self.dropout_rate = dropout_rate
        self.model_ = model
        self.history = None
        self.fpr = None
        self.tpr =None
        self.roc_auc = None


    def _build_model(self, input_dim):
        """
        Build and compile the TensorFlow model.

        Parameters
        ----------
        input_dim : int
            Number of input features.

        Returns
        -------
        model : keras.Model
            Compiled Keras model.
        """
        model = keras.Sequential()

        if self.conv_layers is not None:
            model.add(keras.layers.Input(shape=(input_dim, 1)))

            for (fil, ker), pool in zip(self.conv_layers, self.conv_pool_size):
                model.add(keras.layers.Conv1D(filters=fil, kernel_size=ker, activation='relu', kernel_regularizer='l2'))
                model.add(keras.layers.BatchNormalization())
                if pool !=0:
                    model.add(keras.layers.MaxPooling1D(pool_size=pool))

            model.add(keras.layers.GlobalAveragePooling1D())
        else:
            model.add(keras.layers.Input(shape=(input_dim,)))

        if any(n>0 for n in self.hidden_layers):
            for n_nodes, d_rate in zip(self.hidden_layers, self.dropout_rate, strict=True):
                model.add(keras.layers.Dense(n_nodes, activation='relu'))
                model.add(keras.layers.Dropout(d_rate))

        model.add(keras.layers.Dense(1, activation='sigmoid'))

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',  keras.metrics.AUC(name="auc")])
        return model

    def train(self, X, y, validation_size=0.2):
        """
        Train the model on input data X (features) and y (target).

        Parameters
        ----------
        X : array-like or DataFrame
            Feature data for training.
        y : array-like or Series
            Target labels for training.
        validation_size : float, optional (default=0.2)
            Fraction of data to use for validation.

        Returns
        -------
        loss : float
            Final loss value on the validation set.
        acc : float
            Final accuracy on the validation set.
        roc_auc : float
            ROC AUC score on the validation set.
        """
        #X, y = check_X_y(X, y)
        self.input_dim = X.shape[1]
        self.model_ = self._build_model(self.input_dim)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size)

        # Reshape if using CNN
        if self.conv_layers is not None:
            X_train = np.array(X_train).reshape(-1, self.input_dim, 1)
            X_val = np.array(X_val).reshape(-1, self.input_dim, 1)
        else:
            X_train = np.array(X_train)
            X_val = np.array(X_val)

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train.values))
        train_ds = train_ds.batch(self.batch_size).shuffle(buffer_size=len(X_train)).cache().prefetch(tf.data.AUTOTUNE)
        validation_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val.values))
        validation_ds = validation_ds.batch(self.batch_size).shuffle(buffer_size=len(X_val)).cache().prefetch(tf.data.AUTOTUNE)


        early_stop = keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = self.early_stop_patience,
            restore_best_weights=True
        )

        reduce_LR_plateau = keras.callbacks.ReduceLROnPlateau(
            monitor ="val_loss",
            factor = 0.2,
            patience = self.lr_plateau_reduction_patience
        )

        self.history = self.model_.fit(train_ds, epochs=self.epochs, validation_data=validation_ds,
                                        callbacks=[early_stop, reduce_LR_plateau])
        loss, acc, self.roc_auc = self.model_.evaluate(validation_ds)
        
        # To get the roc
        y_val_predict_prob = self.model_.predict(validation_ds, verbose=0)
        self.fpr, self.tpr, _ = roc_curve(y_val.values, y_val_predict_prob)

        return (loss, acc, self.roc_auc)
    
    def plot_accuracy(self):
        """
        Plot training and validation accuracy for each epoch.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure object with the plotted accuracy.
        """
        fig, ax = plt.subplots()
        if self.epochs == 0:
            return fig
        
        fig, ax = plt.subplots()
        ax.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.grid(True)
        ax.legend()
        ax.set_title('Accuracy per Epoch')
        return fig

    def plot_loss(self):
        """
        Plot training and validation loss for each epoch.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure object with the plotted loss.
        """
        fig, ax = plt.subplots()
        if self.epochs == 0:
            return fig
        
        ax.plot(self.history.history['loss'], label='Training Loss')
        ax.plot(self.history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True)
        ax.legend()
        ax.set_title('Loss per Epoch')
        return fig

    def plot_validation_roc(self):
        """
        Plot the ROC curve computed on the validation set.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure object with the plotted ROC curve.
        """
        fig, ax = plt.subplots()
        if self.epochs == 0:
            return fig

        ax.plot(self.fpr, self.tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {self.roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        return fig

    def predict(self, X, verbose=1):
        """
        Predict class labels for the given input data.

        Parameters
        ----------
        X : array-like or tensor
            Input data for which to predict class labels.
        verbose : int, optional (default=1)
            Verbosity mode for the prediction process.

        Returns
        -------
        y_pred : array-like
            Predicted class labels for each sample in X.
        """
        probs = self.predict_prob(X=X, verbose=verbose)
        return (probs > 0.5).astype(int)
    
    def predict_prob(self, X, verbose=1):
        """
        Predict class probabilities for the given input data.

        Parameters
        ----------
        X : array-like or tensor
            Input data for which to predict class probabilities.
        verbose : int, optional (default=1)
            Verbosity mode for the prediction process.

        Returns
        -------
        probs : array-like
            Predicted class probabilities for each sample in X.
        """

        probs = self.model_.predict(X, verbose=verbose)
        return probs

    def save_model(self, filepath):
        """
        Save the trained model to the specified filepath.

        Parameters
        ----------
        filepath : str
            Path where the model will be saved.

        Raises
        ------
        ValueError
            If the model has not been built or trained yet.
        """
        if self.model_ is not None:
            self.model_.save(filepath)
        else:
            raise ValueError("Model has not been built or trained yet.")

    def set_params(self, **params):
        """
        Set the parameters of the classifier.

        Parameters
        ----------
        **params : dict
            Model parameters to set as attributes.

        Returns
        -------
        self : object
            Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
