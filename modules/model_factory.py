from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor
)

from sklearn.svm import (
    SVC,
    SVR
)

from xgboost import (
    XGBClassifier,
    XGBRegressor
)

from lightgbm import (
    LGBMClassifier,
    LGBMRegressor
)

from catboost import (
    CatBoostClassifier,
    CatBoostRegressor
)


CLASSIFICATION_MODELS = {

    "Logistic Regression":
        LogisticRegression,

    "Decision Tree Classifier":
        DecisionTreeClassifier,

    "Random Forest Classifier":
        RandomForestClassifier,

    "KNN Classifier":
        KNeighborsClassifier,

    "SVM Classifier":
        SVC,

    "AdaBoost Classifier":
        AdaBoostClassifier,

    "XGBoost Classifier":
        XGBClassifier,

    "LightGBM Classifier":
        LGBMClassifier,

    "CatBoost Classifier":
        CatBoostClassifier
}


REGRESSION_MODELS = {

    "Linear Regression":
        LinearRegression,

    "Decision Tree Regressor":
        DecisionTreeRegressor,

    "Random Forest Regressor":
        RandomForestRegressor,

    "KNN Regressor":
        KNeighborsRegressor,

    "SVR":
        SVR,

    "AdaBoost Regressor":
        AdaBoostRegressor,

    "XGBoost Regressor":
        XGBRegressor,

    "LightGBM Regressor":
        LGBMRegressor,

    "CatBoost Regressor":
        CatBoostRegressor
}


def get_model(
        model_name,
        params,
        problem_type
):

    if problem_type == "Classification":

        model_class = CLASSIFICATION_MODELS.get(
            model_name
        )

    else:

        model_class = REGRESSION_MODELS.get(
            model_name
        )

    if model_class is None:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    # XGBoost

    if "XGBoost" in model_name:

        return model_class(
            
            n_jobs=-1,
            
            tree_method="hist",
            
            verbosity=0,
            
            use_label_encoder=False,
            
            eval_metric="mlogloss",
            
            **params
        )

    # LightGBM

    if "LightGBM" in model_name:

        return model_class(

            verbosity=-1,

            random_state=42,

            **params
        )

    # CatBoost

    if "CatBoost" in model_name:

        return model_class(

            verbose=0,

            allow_writing_files=False,

            random_state=42,

            **params
        )

    # Logistic Regression

    if model_name == "Logistic Regression":

        return model_class(

            random_state=42,

            **params
        )

    # Decision Tree

    if "Decision Tree" in model_name:

        return model_class(

            random_state=42,

            **params
        )

    # Random Forest

    if "Random Forest" in model_name:

        return model_class(

            random_state=42,

            n_jobs=-1,

            **params
        )

    # AdaBoost

    if "AdaBoost" in model_name:

        return model_class(

            random_state=42,

            **params
        )

    # Default Models
    # KNN, SVM, SVR, Linear Regression

    return model_class(
        **params
    )