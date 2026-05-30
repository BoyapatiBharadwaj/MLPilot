import io
import joblib
import pandas as pd

def export_model(
        pipeline
):

    if pipeline is None:

        raise ValueError(
            "Pipeline is None."
        )

    buffer = io.BytesIO()

    joblib.dump(
        pipeline,
        buffer
    )

    buffer.seek(0)

    return buffer

def export_csv(
        dataframe
):

    if dataframe is None:

        raise ValueError(
            "Dataframe is None."
        )

    csv = dataframe.to_csv(
        index=False
    )

    return csv.encode("utf-8")

def experiment_to_dataframe(
        comparison_results
):

    return pd.DataFrame(
        comparison_results
    )
    
