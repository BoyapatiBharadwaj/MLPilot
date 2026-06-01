import os
import streamlit as st
import pandas as pd
import uuid



from modules.data_loader import (
    load_dataset,
    get_missing_value_report,
    detect_problem_type
)

from sklearn.metrics import classification_report

from modules.preprocessing import (
    build_preprocessor
)

from modules.evaluator import (
    classification_metrics,
    regression_metrics
)

from modules.hyperparameters import (
    CLASSIFICATION_PARAMS,
    REGRESSION_PARAMS
)

from modules.session_manager import (
    initialize_session
)

import streamlit.components.v1 as components


logo_path = os.path.abspath(
    "MLPilot_Logo.webp"
)

header_path = os.path.abspath(
    "Header_image.webp"
)


def section_card(title):

    st.markdown(
        f"""
        <div style="
        background:white;
        padding:12px 20px;
        border-radius:12px;
        margin:8px 0;
        box-shadow:0 2px 8px rgba(0,0,0,0.05);
        ">
            <h2 style="
            margin:0;
            font-size:34px;
            ">
            {title}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(
    page_title="MLPilot",
    page_icon="MLPilot_Logo.webp",
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

        if (window.clarity) {{

            window.clarity(
                "event",
                "{event_name}"
            );

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

/* Metric Cards */

[data-testid="metric-container"]{

    background:white;

    border-radius:18px;

    padding:18px;

    border:1px solid #E2E8F0;

    box-shadow:0 4px 12px rgba(
        0,
        0,
        0,
        0.08
    );
}

[data-testid="metric-container"] label{

    font-size:14px;

    font-weight:600;
}

[data-testid="metric-container"] > div{

    color:#0F172A;
}

/* DataFrames */

[data-testid="stDataFrame"]{

    border-radius:18px;
}

/* Buttons */

div.stButton > button{

    border-radius:15px !important;
}

/* Reduce Spacing */

div[data-testid="stVerticalBlock"]{

    gap:0.5rem;
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

st.markdown(f"""
<div style="
background:white;
padding:20px 30px;
border-radius:20px;
box-shadow:0 4px 15px rgba(0,0,0,0.08);
margin-bottom:20px;
">

<div style="
display:flex;
align-items:center;
justify-content:space-around;
">

<div style="
display:flex;
align-items:center;
gap:30px;
width:65%;
">

<img src="file:///{logo_path}"
style="
width:420px;
height:auto;
object-fit:contain;
">

<div>

<h1 style="
margin:0;
font-size:54px;
color:#0F172A;
">
MLPilot
</h1>

<p style="
font-size:22px;
color:#2563EB;
margin:0;
font-weight:600;
">
AI-Powered AutoML Platform
</p>

<p style="
font-size:16px;
color:#64748B;
margin-top:8px;
">
Build • Train • Analyze • Deploy
</p>

</div>

</div>

<div style="
width:35%;
display:flex;
justify-content:center;
align-items:center;
">

<img src="file:///{header_path}"
style="
width:720px;
max-width:100%;
height:auto;
">

</div>

</div>

</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown("""
    <div style="
    background:white;
    padding:25px;
    border-radius:20px;
    text-align:center;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
    ">

    <h4>📦 Models</h4>

    <h1 style="
    color:#6366F1;
    ">
    15+
    </h1>

    <p>Built-in Algorithms</p>

    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown("""
    <div style="
    background:white;
    padding:25px;
    border-radius:20px;
    text-align:center;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
    ">

    <h4>🎯 Tasks</h4>

    <h1 style="
    color:#10B981;
    ">
    2
    </h1>

    <p>Classification & Regression</p>

    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown("""
    <div style="
    background:white;
    padding:25px;
    border-radius:20px;
    text-align:center;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
    ">

    <h4>📄 Reports</h4>

    <h1 style="
    color:#F59E0B;
    ">
    PDF
    </h1>

    <p>Detailed Analysis</p>

    </div>
    """, unsafe_allow_html=True)

with c4:

    st.markdown("""
    <div style="
    background:white;
    padding:25px;
    border-radius:20px;
    text-align:center;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
    ">

    <h4>☁ Export</h4>

    <h1 style="
    color:#2563EB;
    ">
    PKL
    </h1>

    <p>Model Export</p>

    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

initialize_session()

if "visitor_id" not in st.session_state:

    st.session_state.visitor_id = (
        str(uuid.uuid4())
    )

    clarity_event(
        "new_visitor"
    )

if "homepage_visit" not in st.session_state:

    clarity_event(
        "homepage_visit"
    )

    st.session_state.homepage_visit = True
    
    

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
    
    

st.sidebar.image(
    "MLPilot_Logo.webp",
    width=90
)

st.sidebar.markdown("""
<h2 style="
margin-bottom:0px;
">
MLPilot
</h2>

<p style="
color:#64748B;
font-size:16px;
">
Build • Train • Analyze • Deploy
</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<h3>
📂 DATASET
</h3>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"],
    label_visibility="collapsed"
)

if uploaded_file is None:

    st.session_state.dataset = None
    st.session_state.results = None
    st.session_state.comparison_results = []
    st.session_state.current_model = None
    st.session_state.current_problem_type = None

    if "dataset_name" in st.session_state:
        del st.session_state.dataset_name

    if "dataset_uploaded_logged" in st.session_state:
        del st.session_state.dataset_uploaded_logged

st.sidebar.markdown("---")

st.sidebar.markdown("""
<h3>
⚡ WORKFLOW
</h3>
""", unsafe_allow_html=True)

workflow_steps = [
    "Upload Dataset",
    "Train Model",
    "Compare Models",
    "View Results",
    "Export"
]

for i, step in enumerate(workflow_steps):

    st.sidebar.markdown(
        f"""
        <div style="
        background:#EFF6FF;
        padding:12px;
        border-radius:12px;
        margin-bottom:10px;
        border-left:4px solid #2563EB;
        ">

        <b>{i+1}. {step}</b>

        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="
background:#EFF6FF;
padding:20px;
border-radius:15px;
border:1px solid #E2E8F0;
">

<h4>
🚀 Get Started
</h4>

<p style="
font-size:14px;
color:#64748B;
">
Upload a dataset and build machine learning models in minutes.
</p>

</div>
""", unsafe_allow_html=True)

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

        rows = len(df)
        cols = len(df.columns)

        if rows < 100:
            clarity_event("rows_under_100")

        elif rows < 1000:
            clarity_event("rows_100_to_1000")

        elif rows < 10000:
            clarity_event("rows_1000_to_10000")

        else:
            clarity_event("rows_over_10000")

        if cols < 10:
            clarity_event("columns_under_10")

        elif cols < 50:
            clarity_event("columns_10_to_50")

        elif cols < 100:
            clarity_event("columns_50_to_100")

        else:
            clarity_event("columns_over_100")

        st.session_state.dataset_uploaded_logged = True

    st.session_state.dataset_name = (
        uploaded_file.name
        .replace(".csv", "")
        .replace(" ", "_")
    )
    
    
if st.session_state.dataset is None:

    st.markdown("""
    <div style="
    background:white;
    padding:50px;
    border-radius:25px;
    box-shadow:0 6px 20px rgba(0,0,0,0.08);
    border:2px dashed #BFDBFE;
    text-align:center;
    margin-top:10px;
    ">

    <div style="
    font-size:60px;
    margin-bottom:10px;
    ">
    ☁️
    </div>

    <h1 style="
    color:#0F172A;
    margin-bottom:10px;
    ">
    Upload Your Dataset to Begin
    </h1>

    <p style="
    font-size:18px;
    color:#64748B;
    margin-bottom:25px;
    ">
    Upload your dataset using the sidebar.
    Start building machine learning models without writing code.
    </p>

    <div style="
    display:flex;
    justify-content:center;
    gap:20px;
    flex-wrap:wrap;
    ">

    <div style="
    background:#EEF2FF;
    padding:15px 25px;
    border-radius:12px;
    ">
    📊 Data Analysis
    </div>

    <div style="
    background:#ECFDF5;
    padding:15px 25px;
    border-radius:12px;
    ">
    🤖 AutoML
    </div>

    <div style="
    background:#FEF3C7;
    padding:15px 25px;
    border-radius:12px;
    ">
    📈 Visualization
    </div>

    <div style="
    background:#DBEAFE;
    padding:15px 25px;
    border-radius:12px;
    ">
    📄 PDF Reports
    </div>

    </div>

    </div>
    """, unsafe_allow_html=True)

    st.stop()
    
  
df = st.session_state.dataset

section_card("📊 Dataset Overview")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Rows",
    f"{df.shape[0]:,}"
)

c2.metric(
    "Columns",
    df.shape[1]
)

c3.metric(
    "Missing",
    int(df.isnull().sum().sum())
)

c4.metric(
    "Duplicates",
    int(df.duplicated().sum())
)

c5.metric(
    "Memory",
    f"{round(df.memory_usage().sum()/1024,1)} KB"
)

section_card("📊 Dataset Explorer")

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Preview",
    "📊 Statistics",
    "⚠ Missing Values",
    "🧬 Data Types"
])


with tab1:

    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=400
    )

    st.write("Shape:", df.shape)
    

with tab2:

    st.info(
        "Summary statistics of numerical and categorical features."
    )
    stats_df = df.describe(
        include="all"
    ).astype(str)

    st.dataframe(
        stats_df,
        use_container_width=True
    )
        

with tab3:

    missing_count = (
        df.isnull().sum().sum()
    )

    if missing_count == 0:

        st.success(
            "No missing values found."
        )

    else:

        st.warning(
            f"{missing_count} missing values detected."
        )
    st.dataframe(
        get_missing_value_report(df)
    )


with tab4:

    type_counts = (
        df.dtypes
        .astype(str)
        .value_counts()
    )

    st.bar_chart(type_counts)
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str)
        })
    )

section_card("🎯 Feature Selection")

left_panel, right_panel = st.columns(
    [2,1]
)

target_column = st.selectbox(
    "Target Column",
    df.columns
)

if (
    "last_target"
    not in st.session_state
    or
    st.session_state.last_target
    != target_column
):

    clarity_event(
        f"target_{target_column}"
    )

    st.session_state.last_target = (
        target_column
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

current_problem = (
    problem_type.lower()
)

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


section_card("⚙ Preprocessing")


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
            .replace(" ", "_")

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
    "last_scaler"
    not in st.session_state
    or
    st.session_state.last_scaler
    != scaling_method
):

    clarity_event(

        scaling_method
            .lower()
            .replace(" ", "_")

    )

    st.session_state.last_scaler = (
        scaling_method
    )


section_card("🤖 Model Selection")


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
        
        
section_card("🚀 Training Configuration")


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
    
        from modules.trainer import train_model
    
        clarity_event(
            "train_button_clicked"
        )

        clarity_event(
            f"trained_{selected_model}"
        )

        clarity_event(
            f"encoding_{encoding_method}"
        )

        clarity_event(
            f"scaler_{scaling_method}"
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

        if results["training_time"] < 1:

            clarity_event(
                "training_under_1s"
            )

        elif results["training_time"] < 5:

            clarity_event(
                "training_under_5s"
            )

        elif results["training_time"] < 30:

            clarity_event(
                "training_under_30s"
            )

        else:

            clarity_event(
                "training_over_30s"
            )
        
if st.session_state.results:
    
    from modules.report_generator import create_report
    
    from modules.visualizer import (
        plot_confusion_matrix,
        plot_roc_curve,
        plot_pr_curve,
        plot_actual_vs_predicted,
        plot_residuals,
        plot_coefficients
    )
    
    results = st.session_state.results
    section_card("📋 Training Summary")
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
    
    
    section_card("📈 Evaluation")
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
    
    
    from modules.exporter import (
        export_model,
        export_csv
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
    
    
    section_card("⬇ Downloads")

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
    
    section_card("📊 Visualizations")

    left, right = st.columns(2)

    if st.session_state.current_problem_type == "Classification":

        cm_fig = plot_confusion_matrix(
            y_test,
            predictions
        )

        roc_fig = plot_roc_curve(
            y_test,
            results["probabilities"]
        )

        with left:

            st.plotly_chart(
                cm_fig,
                use_container_width=True
            )

        if roc_fig:

            with right:

                st.plotly_chart(
                    roc_fig,
                    use_container_width=True
                )

        pr_fig = plot_pr_curve(
            y_test,
            results["probabilities"]
        )

        left2, right2 = st.columns(2)

        if pr_fig:

            with left2:

                st.plotly_chart(
                    pr_fig,
                    use_container_width=True
                )

        try:

            coef_fig = plot_coefficients(
                results["fitted_model"],
                results["feature_names"]
            )

            if coef_fig:

                with right2:

                    st.plotly_chart(
                        coef_fig,
                        use_container_width=True
                    )

        except Exception:
            pass
            
    else:

        left, right = st.columns(2)

        actual_pred_fig = plot_actual_vs_predicted(
            y_test,
            predictions
        )

        with left:

            st.plotly_chart(
                actual_pred_fig,
                use_container_width=True
            )

        residual_fig = plot_residuals(
            y_test,
            predictions
        )

        with right:

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
    
    if (
        "comparison_logged"
        not in st.session_state
    ):

        clarity_event(
            "comparison_viewed"
        )

        st.session_state.comparison_logged = True

    section_card("🏆 Model Comparison")

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
margin-top:50px;
padding:30px 40px;
border-radius:20px;
background:linear-gradient(135deg,#0F172A,#1E293B);
color:white;
box-shadow:0 8px 25px rgba(0,0,0,0.15);
">

<div style="
display:flex;
justify-content:space-between;
align-items:center;
flex-wrap:wrap;
">

<div>

<h2 style="
margin:0;
font-size:32px;
font-weight:700;
color:white;
">
MLPilot
</h2>

<p style="
margin-top:8px;
font-size:16px;
color:#CBD5E1;
">
AI-Powered AutoML Platform
</p>

<p style="
margin-top:5px;
font-size:14px;
color:#94A3B8;
">
Build • Train • Analyze • Deploy
</p>

</div>

<div style="
text-align:right;
">

<p style="
margin:0;
font-size:15px;
color:#CBD5E1;
">
Built with
Python • Scikit-Learn • Streamlit
</p>

<p style="
margin-top:8px;
font-size:14px;
color:#94A3B8;
">
Developed by
<b style="color:white;">
Bharadwaj Boyapati
</b>
</p>

</div>

</div>

<hr style="
border:none;
height:1px;
background:#334155;
margin:20px 0;
">

<div style="
display:flex;
justify-content:space-between;
align-items:center;
flex-wrap:wrap;
font-size:13px;
color:#94A3B8;
">

<div>
© 2026 MLPilot. All Rights Reserved.
</div>

<div>
AutoML • Classification • Regression • Analytics
</div>

</div>

</div>
""", unsafe_allow_html=True)