import time

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold
)

from sklearn.pipeline import Pipeline

from .model_factory import get_model

from .feature_utils import \
    get_feature_names
    
from sklearn.preprocessing import LabelEncoder


def train_model(
        X,
        y,
        model_name,
        model_params,
        problem_type,
        preprocessor,
        test_size,
        random_state,
        cv_folds
):

    start_time = time.time()

    stratify = None

    if problem_type == "Classification":

        class_counts = y.value_counts()

        if class_counts.min() >= 2:

            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=test_size,

        random_state=random_state,

        stratify=stratify
    )

    model = get_model(

        model_name,

        model_params,

        problem_type
    )

    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ])

    label_encoder = None

    if (
        problem_type == "Classification"
        and "XGBoost" in model_name
    ):

        label_encoder = LabelEncoder()

        y_train = label_encoder.fit_transform(
            y_train
        )

        y_test = label_encoder.transform(
            y_test
        )

    pipeline.fit(
        X_train,
        y_train
    )
    
    fitted_model = pipeline.named_steps[
        "model"
    ]
    
    try:

        feature_names = get_feature_names(
            pipeline.named_steps[
                "preprocessor"
            ]
        )

    except Exception:

        feature_names = X.columns.tolist()

    predictions = pipeline.predict(
        X_test
    )

    if label_encoder is not None:

        predictions = label_encoder.inverse_transform(
            predictions.astype(int)
        )

        y_test = label_encoder.inverse_transform(
            y_test.astype(int)
        )

    probabilities = None

    try:

        probabilities = pipeline.predict_proba(
            X_test
        )

    except Exception as e:

        probabilities = None
        
    cv_y = y

    if (
        problem_type == "Classification"
        and "XGBoost" in model_name
    ):

        le_cv = LabelEncoder()

        cv_y = le_cv.fit_transform(y)

    cv_score = None

    try:

        if problem_type == "Classification":

            min_class_size = y.value_counts().min()

            if min_class_size >= 2:

                safe_folds = min(
                    cv_folds,
                    min_class_size
                )

                cv = StratifiedKFold(
                    n_splits=safe_folds,
                    shuffle=True,
                    random_state=random_state
                )

                scoring = "accuracy"

            else:

                cv_score = None
                raise ValueError(
                    "Not enough samples per class for StratifiedKFold."
                )

        else:

            safe_folds = min(
                cv_folds,
                len(X)
            )

            cv = KFold(

                n_splits=max(2, safe_folds),

                shuffle=True,

                random_state=random_state
            )

            scoring = "r2"

        cv_score = cross_val_score(

            pipeline,

            X,

            cv_y,

            scoring=scoring,

            cv=cv

        ).mean()

    except Exception as e:

        print(
            f"CV Error: {e}"
        )

        cv_score = None

    train_time = (
        time.time() - start_time
    )

    return {

        "pipeline": pipeline,
        
        "label_encoder": label_encoder,

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "predictions": predictions,

        "probabilities": probabilities,

        "training_time": train_time,
        
        "fitted_model": fitted_model,

        "cv_score": cv_score,

        "feature_count": X.shape[1],
        
        "feature_names": feature_names,

        "sample_count": X.shape[0]
    }