import pandas as pd
import numpy as np


def safe_round(value, digits=4):

    try:
        return round(float(value), digits)

    except Exception:
        return value


def is_classification_target(y):

    if y.dtype == "object":
        return True

    unique_values = y.nunique()

    if unique_values <= 20:
        return True

    return False


def get_feature_types(df):

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric_columns, categorical_columns


def clean_column_names(df):

    df.columns = [

        str(col)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")

        for col in df.columns
    ]

    return df