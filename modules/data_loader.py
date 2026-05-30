import pandas as pd


def load_dataset(uploaded_file):

    df = pd.read_csv(uploaded_file)

    return df


def dataset_summary(df):

    summary = {

        "rows": df.shape[0],

        "columns": df.shape[1],

        "column_names": df.columns.tolist(),

        "dtypes": df.dtypes.astype(str).to_dict(),

        "missing_values": df.isnull().sum().to_dict(),

        "numeric_columns":
            df.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist(),

        "categorical_columns":
            df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()
    }

    return summary


def get_missing_value_report(df):

    report = pd.DataFrame({

        "Column": df.columns,

        "Missing Count": df.isnull().sum(),

        "Missing Percentage":
            (
                df.isnull().sum()
                /
                len(df)
            ) * 100

    })

    return report.sort_values(
        by="Missing Count",
        ascending=False
    )


def get_target_distribution(df, target):

    return (
        df[target]
        .value_counts()
        .reset_index()
    )


def get_numeric_summary(df):

    return df.describe().T


def detect_problem_type(y):

    if y.dtype == "object":

        return "Classification"

    unique_count = y.nunique()

    if unique_count <= 20:

        return "Classification"

    return "Regression"