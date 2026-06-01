import streamlit as st
import pandas as pd

from modules.data_loader import (
    load_dataset,
    get_missing_value_report,
    detect_problem_type
)

from sklearn.metrics import classification_report

from modules.preprocessing import (
    build_preprocessor
)

from modules.trainer import (
    train_model
)

from modules.evaluator import (
    classification_metrics,
    regression_metrics
)

from modules.hyperparameters import (
    CLASSIFICATION_PARAMS,
    REGRESSION_PARAMS
)

from modules.visualizer import *

from modules.exporter import *

from modules.report_generator import *

from modules.session_manager import (
    initialize_session
)

from modules.visualizer import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_actual_vs_predicted,
    plot_residuals
)

from modules.exporter import (
    export_model,
    export_csv
)

import streamlit.components.v1 as components


st.set_page_config(
    page_title="MLPilot",
    page_icon="MLPilot_Logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

components.html(
    """
    <script type="text/javascript">
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){
            (c[a].q=c[a].q||[]).push(arguments)
        };
        t=l.createElement(r);
        t.async=1;
        t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];
        y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "x082turlkt");
    </script>
    """,
    height=0
)

def clarity_event(event_name):

    components.html(
        f"""
        <script>
        if (typeof clarity !== 'undefined') {{
            clarity("event", "{event_name}");
        }}
        </script>
        """,
        height=0
    )

st.markdown("""
<style>

div.stButton > button {
    width: 100%;
    height: 65px;

    border-radius: 15px !important;

    background: linear-gradient(
        90deg,
        #2563EB,
        #06B6D4
    ) !important;

    color: white !important;

    font-size: 22px !important;

    font-weight: 700 !important;

    border: none !important;

    transition: all 0.3s ease;
}

div.stButton > button p {
    color: white !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}

div.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #1D4ED8,
        #0891B2
    ) !important;

    color: white !important;

    transform: translateY(-2px);

    box-shadow: 0 8px 20px rgba(
        37,
        99,
        235,
        0.35
    );
}

div.stButton > button:hover p {
    color: white !important;
}

div.stButton > button:focus {
    color: white !important;
}

div.stButton > button:active {
    transform: scale(0.98);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

.block-container{
    padding-top:1rem;
}

[data-testid="stSidebar"]{
    background-color:#F8FAFC;
}

h1{
    font-weight:700;
}

</style>
""",
unsafe_allow_html=True)

hide_streamlit_style = """
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
"""

st.markdown(
    hide_streamlit_style,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1,6])

with col1:
    st.image(
        "MLPilot_Logo.png",
        width = 420
    )

with col2:

    st.markdown("""
    <h1 style='margin-bottom:0px;'>
    MLPilot
    </h1>

    <h3>
    AI-Powered AutoML Platform
    </h3>

    <p style='font-size:18px;'>

    Build, train, evaluate and deploy machine learning models
    without writing code.

    </p>

    """,
    unsafe_allow_html=True
    )
    
st.markdown("---")

st.markdown("""
<div style="
padding:25px;
border-radius:15px;
background:linear-gradient(90deg,#0F172A,#1E40AF);
color:white;
margin-top:10px;
margin-bottom:20px;
">

<h2 style="color:white;">
🚀 Build Machine Learning Models in Minutes
</h2>

<p style="font-size:18px;">
Upload datasets, preprocess data, train powerful ML models,
compare results and export production-ready models.
</p>

<p>
✅ Classification & Regression<br>
✅ Hyperparameter Tuning<br>
✅ Interactive Visualizations<br>
✅ Model Comparison Dashboard<br>
✅ PDF Reports & Model Export
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("---")

initialize_session()

if "initialized" not in st.session_state:

    st.session_state.results = None

    st.session_state.dataset = None

    st.session_state.comparison_results = []

    st.session_state.current_model = None

    st.session_state.current_problem_type = None

    st.session_state.initialized = True



if "results" not in st.session_state:
    st.session_state.results = None

if "dataset" not in st.session_state:
    st.session_state.dataset = None
    
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = []
    
    

col1, col2 = st.sidebar.columns([1,4])

with col1:
    st.image(
        "MLPilot_Logo.png",
        width=150
    )

with col2:
    st.markdown(
        "### MLPilot"
    )

st.sidebar.caption(
    "Build • Train • Analyze • Deploy"
)

st.sidebar.markdown("---")

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)


if uploaded_file is not None:

    df = load_dataset(uploaded_file)

    st.session_state.dataset = df
    
    if (
        "dataset_uploaded_logged"
        not in st.session_state
    ):

        clarity_event(
            "dataset_uploaded"
        )

        clarity_event(
            st.session_state.dataset_name
        )

        st.session_state.dataset_uploaded_logged = True

    st.session_state.dataset_name = (
        uploaded_file.name
        .replace(".csv", "")
        .replace(" ", "_")
    )
    
    
if st.session_state.dataset is None:

    st.info("Upload dataset to begin")

    st.stop()
    
    
df = st.session_state.dataset

if "dataset_size_logged" not in st.session_state:

    clarity_event(
        f"rows_{len(df)}"
    )

    clarity_event(
        f"columns_{len(df.columns)}"
    )

    st.session_state.dataset_size_logged = True


st.header("Dataset Explorer")

tab1, tab2, tab3, tab4 = st.tabs([
    "Preview",
    "Statistics",
    "Missing Values",
    "Data Types"
])


with tab1:

    st.dataframe(
        df,
        use_container_width=True
    )

    st.write("Shape:", df.shape)
    

with tab2:

    st.dataframe(
        df.describe(include="all")
    )
    

with tab3:

    st.dataframe(
        get_missing_value_report(df)
    )


with tab4:

    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str)
        })
    )
    

st.header("Feature Selection")


target_column = st.selectbox(
    "Target Column",
    df.columns
)


available_features = [

    col for col in df.columns

    if col != target_column
]


selected_features = st.multiselect(

    "Feature Columns",

    available_features,

    default=available_features
)


X = df[selected_features]

y = df[target_column]

if y.isna().all():

    st.error(
        "Target column contains only missing values."
    )

    st.stop()


detected_type = detect_problem_type(y)


problem_type = st.radio(

    "Problem Type",

    ["Classification", "Regression"],

    index=0 if detected_type ==
    "Classification" else 1
)

current_problem = problem_type.lower()

if (
    "last_problem_type"
    not in st.session_state
    or
    st.session_state.last_problem_type
    != current_problem
):

    clarity_event(
        current_problem
    )

    st.session_state.last_problem_type = (
        current_problem
    )


st.header("Preprocessing")


col1, col2 = st.columns(2)

with col1:

    numeric_strategy = st.selectbox(
        "Numerical Missing Values",
        [
            "mean",
            "median",
            "most_frequent",
            "constant"
        ]
    )

    numeric_constant = 0

    if numeric_strategy == "constant":

        numeric_constant = st.number_input(
            "Numeric Constant Value",
            value=0
        )

with col2:

    categorical_strategy = st.selectbox(
        "Categorical Missing Values",
        [
            "most_frequent",
            "constant"
        ]
    )

    categorical_constant = "missing"

    if categorical_strategy == "constant":

        categorical_constant = st.text_input(
            "Categorical Constant Value",
            value="missing"
        )
        
        
encoding_method = st.selectbox(

    "Encoding Method",

    [
        "One-Hot Encoding",
        "Ordinal Encoding",
        "Target Encoding"
    ]
)

if (
    "last_encoding"
    not in st.session_state
    or
    st.session_state.last_encoding
    != encoding_method
):

    clarity_event(
        encoding_method
            .lower()
            .replace(" ","_")
    )

    st.session_state.last_encoding = (
        encoding_method
    )


scaling_method = st.selectbox(

    "Scaling Method",

    [
        "None",
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler"
    ]
)

if (
    "last_scaling"
    not in st.session_state
    or
    st.session_state.last_scaling
    != scaling_method
):

    clarity_event(
        scaling_method
            .lower()
    )

    st.session_state.last_scaling = (
        scaling_method
    )


st.header("Model Selection")


if problem_type == "Classification":

    model_names = list(
        CLASSIFICATION_PARAMS.keys()
    )

else:

    model_names = list(
        REGRESSION_PARAMS.keys()
    )
    
    
selected_model = st.selectbox(
    "Choose Model",
    model_names
)

current_model_event = (
    selected_model
        .lower()
        .replace(" ", "_")
)

if (
    "last_model_event"
    not in st.session_state
    or
    st.session_state.last_model_event
    != current_model_event
):

    clarity_event(current_model_event)

    st.session_state.last_model_event = (
        current_model_event
    )


st.subheader("Hyperparameters")


if problem_type == "Classification":

    model_params_config = \
        CLASSIFICATION_PARAMS[
            selected_model
        ]

else:

    model_params_config = \
        REGRESSION_PARAMS[
            selected_model
        ]
        
        
user_params = {}


for param, value in model_params_config.items():

    if isinstance(value, list):

        user_params[param] = st.selectbox(
            param,
            value
        )

    elif isinstance(value, bool):

        user_params[param] = st.checkbox(
            param,
            value=value
        )

    elif isinstance(value, int):

        user_params[param] = int(
            st.number_input(
                param,
                value=value,
                step=1
            )
        )

    elif isinstance(value, float):

        user_params[param] = st.number_input(
            param,
            min_value=0.0,
            value=float(value),
            step=0.01
        )

    else:

        user_params[param] = value
        
        
st.header("Training Configuration")


col1, col2, col3 = st.columns(3)


with col1:

    test_size = st.slider(
        "Test Size",
        0.1,
        0.5,
        0.2
    )
    
    
with col2:

    random_state = int(
        st.number_input(
            "Random State",
            value=42,
            step=1
        )
    )
    
    
with col3:

    cv_folds = st.slider(
        "CV Folds",
        2,
        10,
        5
    )
    
    
train_clicked = st.button(
    "🚀 Train & Evaluate Model",
    use_container_width=True
)


if train_clicked:
    
        clarity_event(
            "train_button_clicked"
        )
        
        clarity_event(
            f"cv_folds_{cv_folds}"
        )

        clarity_event(
            f"test_size_{test_size}"
        )

        clarity_event(
            f"features_{len(selected_features)}"
        )
    
        if len(selected_features) == 0:

            st.error(
                "Select at least one feature."
            )

            st.stop()
            
        preprocessor, _, _ = \
            build_preprocessor(

                X,
                
                y,

                numeric_strategy,

                categorical_strategy,

                numeric_constant,

                categorical_constant,

                encoding_method,

                scaling_method
            )
            
        progress = st.progress(0)

        progress.progress(25)
        
        try:

            results = train_model(

                X=X,

                y=y,

                model_name=selected_model,

                model_params=user_params,

                problem_type=problem_type,

                preprocessor=preprocessor,

                test_size=test_size,

                random_state=random_state,

                cv_folds=cv_folds
            )

        except Exception as e:
            
            clarity_event(
                "training_failed"
            )

            st.error(
                f"Training Failed: {str(e)}"
            )

            st.stop()
        
        comparison_entry = {

            "Model": selected_model,

            "CV Score": results["cv_score"],

            "Training Time":
                results["training_time"]
        }

        st.session_state.comparison_results.append(
            comparison_entry
        )
        
        
        progress.progress(75)
        st.session_state.results = results
        st.session_state.current_model = selected_model
        st.session_state.current_problem_type = problem_type
        progress.progress(100)
        
        clarity_event(
            "training_success"
        )
        
        st.success(
            "Training Completed"
        )
        
if st.session_state.results:
    results = st.session_state.results
    st.header("Training Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Samples",
        results["sample_count"]
    )

    c2.metric(
        "Features",
        results["feature_count"]
    )

    c3.metric(
        "Training Time",
        f'{results["training_time"]:.2f}s'
    )

    c4.metric(
        "CV Score",

        round(
            results["cv_score"],
            4
        )

        if results["cv_score"] is not None

        else "N/A"
    )
    
    
    st.header("Evaluation")
    y_test = results["y_test"]

    predictions = results["predictions"]
    
    if (
        st.session_state.current_problem_type
        == "Classification"
    ):
        metrics = classification_metrics(
            y_test,
            predictions,
            results["probabilities"]
        )
        
        clarity_event(
            "classification_evaluated"
        )
        
        metric_df = pd.DataFrame({

            "Metric": [

                "Accuracy",

                "Precision",

                "Recall",

                "F1 Score"

            ],

            "Value": [

                metrics["Accuracy"],

                metrics["Precision"],

                metrics["Recall"],

                metrics["F1 Score"]

            ]
        })
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{metrics['Accuracy']:.4f}"
        )

        c2.metric(
            "Precision",
            f"{metrics['Precision']:.4f}"
        )

        c3.metric(
            "Recall",
            f"{metrics['Recall']:.4f}"
        )

        c4.metric(
            "F1 Score",
            f"{metrics['F1 Score']:.4f}"
        )

        if metrics["ROC-AUC"] is not None:

            st.metric(
                "ROC-AUC",
                f"{metrics['ROC-AUC']:.4f}"
            )
        
        st.subheader(
            "Classification Report"
        )

        report_df = pd.DataFrame(
            classification_report(
                y_test,
                predictions,
                output_dict=True
            )
        ).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )
                
    else:
        metrics = regression_metrics(

            y_test,

            predictions
        )
        
        clarity_event(
            "regression_evaluated"
        )
        
        metric_df = pd.DataFrame({

            "Metric":
                list(metrics.keys()),

            "Value":
                list(metrics.values())
        })
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "R²",
            f"{metrics['R2 Score']:.4f}"
        )

        c2.metric(
            "MAE",
            f"{metrics['MAE']:.4f}"
        )

        c3.metric(
            "MSE",
            f"{metrics['MSE']:.4f}"
        )

        c4.metric(
            "RMSE",
            f"{metrics['RMSE']:.4f}"
        )

        c5.metric(
            "MAPE",
            f"{metrics['MAPE']:.4f}"
        )
            
    csv_data = export_csv(df)
    
    dataset_name = uploaded_file.name
    
    report_buffer = create_report(

        st.session_state.current_model,

        dataset_name,

        st.session_state.current_problem_type,

        metrics,

        results["training_time"],

        results["cv_score"],

        user_params,

        results["sample_count"],

        results["feature_count"]
    )
        
    
    model_name_clean = (
        st.session_state.current_model
            .replace(" ", "_")
    )

    dataset_name = st.session_state.get(
        "dataset_name",
        "dataset"
    )
    
    
    st.header(
        "Downloads"
    )

    model_buffer = export_model(
        results["pipeline"]
    )

    d1, d2, d3 = st.columns(3)

    with d1:

        model_download = st.download_button(
            "⬇ Download Model",
            model_buffer,
            f"{model_name_clean}.pkl"
        )

        if model_download:

            clarity_event(
                "model_downloaded"
            )

    with d2:

        dataset_download = st.download_button(
            "⬇ Download Dataset",
            csv_data,
            f"{dataset_name}.csv"
        )

        if dataset_download:

            clarity_event(
                "dataset_downloaded"
            )

    with d3:

        report_download = st.download_button(
            "⬇ Download Report",
            report_buffer,
            f"{model_name_clean}_Report.pdf"
        )

        if report_download:

            clarity_event(
                "report_downloaded"
            )
            
    if (
        "visualization_logged"
        not in st.session_state
    ):

        clarity_event(
            "visualizations_viewed"
        )

        st.session_state.visualization_logged = True
    
    st.header("Visualizations")

    if st.session_state.current_problem_type == "Classification":

        cm_fig = plot_confusion_matrix(
            y_test,
            predictions
        )

        st.plotly_chart(
            cm_fig,
            use_container_width=True
        )

        roc_fig = plot_roc_curve(
            y_test,
            results["probabilities"]
        )

        if roc_fig:
            st.plotly_chart(
                roc_fig,
                use_container_width=True
            )

        pr_fig = plot_pr_curve(
            y_test,
            results["probabilities"]
        )
        
        try:

            coef_fig = plot_coefficients(
                results["fitted_model"],
                results["feature_names"]
            )

            if coef_fig:

                st.subheader(
                    "Feature Coefficients"
                )

                st.plotly_chart(
                    coef_fig,
                    use_container_width=True
                )

        except Exception:
            pass

        if pr_fig:
            st.plotly_chart(
                pr_fig,
                use_container_width=True
            )
            
    else:

        actual_pred_fig = \
            plot_actual_vs_predicted(
                y_test,
                predictions
            )

        st.plotly_chart(
            actual_pred_fig,
            use_container_width=True
        )

        residual_fig = \
            plot_residuals(
                y_test,
                predictions
            )

        st.plotly_chart(
            residual_fig,
            use_container_width=True
        )
        
    
        
if (
    "comparison_results"
    in st.session_state
    and
    len(
        st.session_state.comparison_results
    ) > 0
):

    st.header(
        "Model Comparison"
    )

    leaderboard = pd.DataFrame(
        st.session_state.comparison_results
    )
    
    best_model = leaderboard.loc[
        leaderboard["CV Score"].idxmax()
    ]

    st.success(
        f"🏆 Best Model: {best_model['Model']} | Score: {best_model['CV Score']:.4f}"
    )

    leaderboard = leaderboard.sort_values(
        "CV Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    leaderboard.insert(
        0,
        "Rank",
        range(
            1,
            len(leaderboard) + 1
        )
    )

    st.dataframe(
        leaderboard.style.highlight_max(
            subset=["CV Score"],
            color="lightgreen"
        ),
        use_container_width=True
    )
    

    
st.markdown("""
<div style="
text-align:center;
padding:35px;
margin-top:20px;
margin-bottom:10px;
border-radius:18px;
background:linear-gradient(135deg,#0F172A,#1E3A8A);
color:white;
box-shadow:0 8px 25px rgba(0,0,0,0.15);
">

<h2 style="
margin-bottom:10px;
color:white;
font-weight:700;
">
MLPilot
</h2>

<p style="
font-size:18px;
color:#E2E8F0;
margin-bottom:15px;
">
AI-Powered AutoML Platform
</p>

<p style="
font-size:16px;
color:#CBD5E1;
margin-bottom:20px;
">
Build • Train • Analyze • Deploy
</p>

<hr style="
border:0;
height:1px;
background:#475569;
margin:15px 0;
">

<p style="
font-size:15px;
color:#94A3B8;
">
Developed by
<b style="color:white;">
Bharadwaj Boyapati
</b>
</p>

</div>
""", unsafe_allow_html=True)