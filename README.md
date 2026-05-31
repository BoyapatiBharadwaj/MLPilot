# 🚀 MLPilot – AI-Powered AutoML Platform

MLPilot is a production-ready AutoML web application that enables users to build, train, evaluate, compare, and export machine learning models without writing code.

Built using **Python**, **Streamlit**, **Scikit-Learn**, **XGBoost**, **LightGBM**, and **CatBoost**, MLPilot provides an intuitive interface for dataset exploration, preprocessing, model training, visualization, and report generation.

---

## 🌟 Features

### 📂 Dataset Upload & Exploration

* Upload CSV datasets
* Dataset preview
* Shape information
* Missing value analysis
* Data type inspection
* Summary statistics
* Feature overview

---

### 🔧 Data Preprocessing

#### Missing Value Handling

Numerical Columns:

* Mean
* Median
* Most Frequent
* Constant Value

Categorical Columns:

* Most Frequent
* Constant Value

#### Encoding Methods

* One-Hot Encoding
* Ordinal Encoding
* Target Encoding

#### Feature Scaling

* None
* StandardScaler
* MinMaxScaler
* RobustScaler

---

### 🎯 Feature Selection

* Select target column
* Select input features
* Remove unwanted features
* Automatic feature count tracking

---

### 🤖 Supported Machine Learning Models

#### Classification

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* KNN Classifier
* SVM Classifier
* AdaBoost Classifier
* XGBoost Classifier
* LightGBM Classifier
* CatBoost Classifier

#### Regression

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* KNN Regressor
* SVR
* AdaBoost Regressor
* XGBoost Regressor
* LightGBM Regressor
* CatBoost Regressor

---

### ⚙ Dynamic Hyperparameter Configuration

MLPilot automatically displays relevant hyperparameters for the selected model.

Examples:

* Decision Tree
* Random Forest
* XGBoost
* LightGBM
* CatBoost
* Logistic Regression
* SVM
* KNN
* AdaBoost

---

### 📊 Model Evaluation

#### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Classification Report

#### Regression Metrics

* R² Score
* MAE
* MSE
* RMSE
* MAPE

---

### 📈 Visualizations

#### Classification

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Feature Importance
* Coefficient Analysis

#### Regression

* Actual vs Predicted
* Residual Plot
* Feature Importance
* Coefficient Analysis

---

### 🏆 Model Comparison Dashboard

Compare multiple trained models using:

| Model | CV Score | Training Time |
| ----- | -------- | ------------- |

* Automatic ranking
* Performance comparison
* Training time analysis

---

### 📄 Export Features

Download:

* Trained Model (.pkl)
* Dataset (.csv)
* Detailed PDF Report

Generated reports include:

* Dataset Information
* Model Information
* Hyperparameters
* Performance Metrics
* Classification Report
* Confusion Matrix
* Training Statistics

---

## 🏗 Project Structure

```text
MLPilot/
│
├── app.py
│
├── assets/
│   └── MLPilot_Logo.png
│
├── modules/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── visualizer.py
│   ├── exporter.py
│   ├── report_generator.py
│   ├── hyperparameters.py
│   ├── model_factory.py
│   ├── feature_utils.py
│   └── session_manager.py
│
├── requirements.txt
│
└── README.md
```

---

## 🛠 Installation

Clone Repository

```bash
git clone https://github.com/your-username/MLPilot.git
```

```bash
cd MLPilot
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run Application

```bash
streamlit run app.py
```

---

## ☁ Deployment

MLPilot is fully deployable on:

* Streamlit Community Cloud
* Hugging Face Spaces
* Render
* Railway
* Docker Environments

No paid services are required.

---

## 🔒 Security & Reliability

* Missing value validation
* Safe cross-validation handling
* Automatic XGBoost label encoding
* Unknown category handling
* Model export compatibility
* Robust error handling

---

## 📌 Tech Stack

Frontend:

* Streamlit

Backend:

* Python

Machine Learning:

* Scikit-Learn
* XGBoost
* LightGBM
* CatBoost

Visualization:

* Plotly
* Matplotlib

Reporting:

* ReportLab

Model Persistence:

* Joblib

---

## 🚀 Future Enhancements

* AutoML Model Search
* SHAP Explainability
* Feature Selection Algorithms
* Deep Learning Models
* Time Series Forecasting
* Hyperparameter Optimization
* Model Deployment API
* User Authentication
* Experiment Tracking

---

## 👨‍💻 Author

Boyapati Bharadwaj

MLPilot – AI-Powered AutoML Platform

Build • Train • Analyze • Deploy

---

## ⭐ Support

If you found this project useful:

* Star the repository
* Fork the project
* Share with the community
* Contribute improvements

Happy Modeling with MLPilot! 🚀
