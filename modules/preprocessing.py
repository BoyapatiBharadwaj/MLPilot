import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

def get_scaler(scaling_method):

    scalers = {
        "None": "passthrough",
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler()
    }

    return scalers.get(
        scaling_method,
        "passthrough"
    )


def get_encoder(encoding_method):

    if encoding_method == "One-Hot Encoding":

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    elif encoding_method == "Ordinal Encoding":

        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    return OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )


def build_preprocessor(
        X,
        numeric_strategy,
        categorical_strategy,
        numeric_constant,
        categorical_constant,
        encoding_method,
        scaling_method
):
    
    if X.shape[1] == 0:

        raise ValueError(
            "No feature columns selected."
        )

    numerical_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if numeric_strategy == "constant":

        numeric_imputer = SimpleImputer(
            strategy="constant",
            fill_value=numeric_constant
        )

    else:

        numeric_imputer = SimpleImputer(
            strategy=numeric_strategy
        )

    if categorical_strategy == "constant":

        categorical_imputer = SimpleImputer(
            strategy="constant",
            fill_value=categorical_constant
        )

    else:

        categorical_imputer = SimpleImputer(
            strategy=categorical_strategy
        )

    numeric_pipeline = Pipeline([
        ("imputer", numeric_imputer),
        ("scaler", get_scaler(scaling_method))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", categorical_imputer),
        ("encoder", get_encoder(encoding_method))
    ])

    transformers = []

    if len(numerical_cols) > 0:

        transformers.append(
            (
                "num",
                numeric_pipeline,
                numerical_cols
            )
        )

    if len(categorical_cols) > 0:

        transformers.append(
            (
                "cat",
                categorical_pipeline,
                categorical_cols
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return (
        preprocessor,
        numerical_cols,
        categorical_cols
    )