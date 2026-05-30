CLASSIFICATION_PARAMS = {

    "Logistic Regression": {
        "C": 1.0,
        "max_iter": 500,
        "solver": ["lbfgs", "liblinear", "saga"]
    },

    "Decision Tree Classifier": {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "splitter": ["best", "random"],
        "max_features": ["sqrt", "log2", "None"]
    },

    "Random Forest Classifier": {
        "n_estimators": 100,
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": ["sqrt", "log2", "None"],
        "bootstrap": [True, False]
    },

    "KNN Classifier": {
        "n_neighbors": 5,
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
    },

    "SVM Classifier": {
        "C": 1.0,
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": ["scale", "auto"]
    },

    "AdaBoost Classifier": {
        "n_estimators": 50,
        "learning_rate": 1.0
    },

    "XGBoost Classifier": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "min_child_weight": 1,
        "gamma": 0.0
    },

    "LightGBM Classifier": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": -1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0
    },

    "CatBoost Classifier": {
        "iterations": 100,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3.0
    }
}


REGRESSION_PARAMS = {

    "Linear Regression": {},

    "Decision Tree Regressor": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": ["sqrt", "log2", "None"]
    },

    "Random Forest Regressor": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": [True, False],
        "max_features": ["sqrt", "log2", "None"]
    },

    "KNN Regressor": {
        "n_neighbors": 5,
        "weights": ["uniform", "distance"]
    },

    "SVR": {
        "C": 1.0,
        "kernel": ["linear", "poly", "rbf"],
        "gamma": ["scale", "auto"]
    },

    "AdaBoost Regressor": {
        "n_estimators": 50,
        "learning_rate": 1.0
    },

    "XGBoost Regressor": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0
    },

    "LightGBM Regressor": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": -1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0
    },

    "CatBoost Regressor": {
        "iterations": 100,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3.0
    }
}