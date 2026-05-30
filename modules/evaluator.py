import numpy as np

from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,

    confusion_matrix,
    classification_report,

    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


def classification_metrics(
        y_true,
        y_pred,
        probabilities=None
):

    metrics = {

        "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "Precision":
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),

        "F1 Score":
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            
        "ROC-AUC": None,

        "Confusion Matrix":
            confusion_matrix(
                y_true,
                y_pred
            ),

        "Classification Report":
            classification_report(
                y_true,
                y_pred
            )
    }
    
    try:

        if probabilities is not None:

            if probabilities.shape[1] == 2:

                metrics["ROC-AUC"] = (
                    roc_auc_score(
                        y_true,
                        probabilities[:, 1]
                    )
                )

            else:

                metrics["ROC-AUC"] = (
                    roc_auc_score(
                        y_true,
                        probabilities,
                        multi_class="ovr"
                    )
                )

    except Exception:

        metrics["ROC-AUC"] = None

    return metrics


def regression_metrics(
        y_true,
        y_pred
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    metrics = {

        "R2 Score":
            r2_score(
                y_true,
                y_pred
            ),

        "MAE":
            mean_absolute_error(
                y_true,
                y_pred
            ),

        "MSE":
            mse,

        "RMSE":
            np.sqrt(mse),

        "MAPE":
            mean_absolute_percentage_error(
                y_true,
                y_pred
            )
    }

    return metrics