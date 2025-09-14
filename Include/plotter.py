import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

def plot_confusion_matrix(cm, labels=["Actual 0", "Actual 1"], pred_labels=["Predicted 0", "Predicted 1"], title="Confusion Matrix"):
    """
    Plots a 2x2 confusion matrix as a quadrant graph with annotations.
    cm: 2x2 array-like (confusion matrix)
    labels: list of actual class labels (y-axis)
    pred_labels: list of predicted class labels (x-axis)
    """
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # Tick marks and labels
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(pred_labels)
    ax.set_yticklabels(labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Annotate each quadrant
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            if i == 0 and j == 0:
                txt = f"TN\n{cm[i, j]}"
            elif i == 0 and j == 1:
                txt = f"FP\n{cm[i, j]}"
            elif i == 1 and j == 0:
                txt = f"FN\n{cm[i, j]}"
            else:
                txt = f"TP\n{cm[i, j]}"
            ax.text(j, i, txt,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    """
    Plot the ROC curve for binary classification.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Target scores (probabilities or confidence values).
    title : str, optional (default="ROC Curve")
        Title for the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object with the plotted ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig