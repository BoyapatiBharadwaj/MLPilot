from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,

    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

import numpy as np


def classification_scores(
        y_true,
        y_pred,
        probabilities=None
):

    scores = {

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

        "F1":
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
    }

    try:

        if probabilities is not None:

            if probabilities.shape[1] == 2:

                scores["ROC_AUC"] = roc_auc_score(
                    y_true,
                    probabilities[:, 1]
                )

    except Exception:

        scores["ROC_AUC"] = None

    return scores


def regression_scores(
        y_true,
        y_pred
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    return {

        "R2":
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