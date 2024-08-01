import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import shap
import numpy as np

# Function to encode non-numeric columns
def encode_columns(df):
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    return df, label_encoders

# Upload CSV
st.title("Dana AI - Root Cause Analysis")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # Encode non-numeric columns
    df, label_encoders = encode_columns(df)
    
    # Select target variable
    target_variable = st.selectbox("Select the target variable", df.columns)

    # Select model
    model_choice = st.selectbox("Select the model", ["AdaBoost", "CatBoost", "DecisionTree"])

    if st.button("Train Model"):
        X = df.drop(columns=[target_variable])
        y = df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "AdaBoost":
            model = AdaBoostClassifier()
        elif model_choice == "CatBoost":
            model = CatBoostClassifier(verbose=0)
        elif model_choice == "DecisionTree":
            model = DecisionTreeClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Store model in session state
        st.session_state['trained_model'] = model
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test

        # Performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Convert classification report to DataFrame
        metrics = {
            "Accuracy": accuracy,
            "AUC": auc,
            "F1 Score": f1
        }

        for key, value in report.items():
            if isinstance(value, dict):
                for metric, score in value.items():
                    metrics[f"{key} {metric}"] = score
            else:
                metrics[key] = value

        report_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

        st.write("Model Performance Metrics:")
        st.write(report_df)

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X.columns
            feature_importances = pd.DataFrame(importances, index=feature_names, columns=["Importance"]).sort_values("Importance", ascending=False)

            fig = px.bar(feature_importances, x=feature_importances.index, y='Importance', title='Feature Importance')
            st.plotly_chart(fig)
