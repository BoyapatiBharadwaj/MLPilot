import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from sklearn.tree import plot_tree

def plot_confusion_matrix(
        y_true,
        y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    labels = sorted(
        list(set(y_true))
    )

    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        aspect="auto"
    )

    fig.update_layout(
        title="Confusion Matrix"
    )

    return fig

def plot_roc_curve(
        y_true,
        probabilities
):

    if probabilities is None:
        return None

    try:
        
        if probabilities.shape[1] != 2:
            return None

        probs = probabilities[:, 1]

        fpr, tpr, _ = roc_curve(
            y_true,
            probs
        )

        fig = px.line(
            x=fpr,
            y=tpr
        )

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )

        return fig

    except Exception:

        return None
    
def plot_pr_curve(
        y_true,
        probabilities
):

    if probabilities is None:
        return None

    try:

        if probabilities.shape[1] != 2:
            return None

        probs = probabilities[:, 1]

        precision, recall, _ = \
            precision_recall_curve(
                y_true,
                probs
            )

        fig = px.line(
            x=recall,
            y=precision
        )

        fig.update_layout(
            title="Precision Recall Curve"
        )

        return fig

    except Exception:

        return None
    
def plot_actual_vs_predicted(
        y_true,
        y_pred
):

    df = pd.DataFrame({

        "Actual": y_true,

        "Predicted": y_pred
    })

    fig = px.scatter(
        df,
        x="Actual",
        y="Predicted"
    )

    fig.update_layout(
        title="Actual vs Predicted"
    )

    return fig

def plot_residuals(
        y_true,
        y_pred
):

    residuals = np.asarray(
        y_true
    ) - np.asarray(
        y_pred
    )

    fig = px.scatter(
        x=y_pred,
        y=residuals
    )

    fig.update_layout(
        title="Residual Plot",
        xaxis_title="Predicted",
        yaxis_title="Residual"
    )

    return fig

def plot_feature_importance(
        model,
        feature_names,
        top_n=20
):

    if not hasattr(
        model,
        "feature_importances_"
    ):
        return None

    importance = model.feature_importances_
    
    if len(feature_names) != len(importance):

        min_len = min(
            len(feature_names),
            len(importance)
        )

        feature_names = feature_names[:min_len]

        importance = importance[:min_len]

    df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance
    })

    df = df.sort_values(
        "Importance",
        ascending=False
    )

    df = df.head(top_n)

    fig = px.bar(

        df,

        x="Importance",

        y="Feature",

        orientation="h"
    )

    fig.update_layout(
        title="Feature Importance"
    )

    return fig

def plot_coefficients(
        model,
        feature_names
):

    if not hasattr(
        model,
        "coef_"
    ):
        return None

    coef = model.coef_

    if len(coef.shape) > 1:
        coef = coef[0]
        
    if len(feature_names) != len(coef):

        min_len = min(
            len(feature_names),
            len(coef)
        )

        feature_names = feature_names[:min_len]

        coef = coef[:min_len]

    df = pd.DataFrame({

        "Feature": feature_names,

        "Coefficient": coef
    })

    df = df.sort_values(
        "Coefficient"
    )

    fig = px.bar(

        df,

        x="Coefficient",

        y="Feature",

        orientation="h"
    )

    fig.update_layout(
        title="Feature Coefficients"
    )

    return fig

def plot_decision_tree_model(
        model,
        feature_names
):

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    plot_tree(

        model,

        feature_names=feature_names,

        filled=True,

        rounded=True,

        fontsize=8,

        ax=ax
    )

    return fig

def get_tree_statistics(
        model
):

    if not hasattr(
        model,
        "tree_"
    ):
        return None

    return {

        "Tree Depth":
            model.tree_.max_depth,

        "Leaf Count":
            model.tree_.n_leaves,

        "Node Count":
            model.tree_.node_count
    }
    
def plot_prediction_distribution(
        predictions
):

    fig = px.histogram(
        x=predictions,
        nbins=30
    )

    fig.update_layout(
        title="Prediction Distribution"
    )

    return fig

