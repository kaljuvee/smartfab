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
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_classif

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
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        
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

    if st.button("Analyze Feature Importance"):
        if 'trained_model' not in st.session_state:
            st.error("Please train the model first.")
        else:
            model = st.session_state['trained_model']
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
            
            st.subheader("Feature Importance Analysis")

            # 1. Built-in Feature Importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = X_train.columns
                feature_importances = pd.DataFrame(importances, index=feature_names, columns=["Importance"]).sort_values("Importance", ascending=False)

                fig = px.bar(feature_importances, x=feature_importances.index, y='Importance', title='Built-in Feature Importance')
                st.plotly_chart(fig)

                st.markdown("""
                **Interpreting Built-in Feature Importance:**
                - X-axis: Feature names
                - Y-axis: Importance score (0 to 1)
                - Higher bars indicate more important features in the model's decision-making process.
                - This method is specific to the model type (e.g., mean decrease in impurity for tree-based models).
                - Features are ranked based on their contribution to the model's predictions.
                """)

            # 2. Permutation Importance
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            perm_importances = pd.DataFrame({'Feature': X_test.columns, 'Importance': perm_importance.importances_mean}).sort_values('Importance', ascending=False)
            
            fig = px.bar(perm_importances, x='Feature', y='Importance', title='Permutation Importance')
            st.plotly_chart(fig)
            st.markdown("""
            **Interpreting Permutation Importance:**
            - X-axis: Feature names
            - Y-axis: Importance score (decrease in model performance when the feature is permuted)
            - Higher bars indicate features that, when randomly shuffled, cause a larger decrease in model performance.
            - This method is model-agnostic and can capture both linear and non-linear relationships.
            - It measures the impact of each feature on model performance, not just its presence in the model.
            """)
            # 3. Recursive Feature Elimination
            rfe = RFE(estimator=model, n_features_to_select=1)
            rfe.fit(X_train, y_train)
            rfe_importances = pd.DataFrame({'Feature': X_train.columns, 'RFE Ranking': rfe.ranking_}).sort_values('RFE Ranking')
            
            fig = px.bar(rfe_importances, x='Feature', y='RFE Ranking', title='Recursive Feature Elimination Ranking')
            st.plotly_chart(fig)
            st.markdown("""
            **Interpreting Recursive Feature Elimination (RFE) Ranking:**
            - X-axis: Feature names
            - Y-axis: RFE Ranking (lower is better)
            - Features with lower ranking (shorter bars) are considered more important.
            - RFE recursively removes features and ranks them based on when they were eliminated.
            - Rank 1 indicates the most important feature, with higher ranks being less important.
            - This method considers feature dependencies and can capture complex relationships.
            """)
            # 4. Mutual Information
            mi_scores = mutual_info_classif(X_train, y_train)
            mi_df = pd.DataFrame({'Feature': X_train.columns, 'Mutual Information': mi_scores}).sort_values('Mutual Information', ascending=False)
            
            fig = px.bar(mi_df, x='Feature', y='Mutual Information', title='Mutual Information Feature Importance')
            st.plotly_chart(fig)
            st.markdown("""
            **Interpreting Mutual Information:**
            - X-axis: Feature names
            - Y-axis: Mutual Information score (0 to 1)
            - Higher bars indicate stronger statistical dependency between the feature and the target variable.
            - Mutual Information measures how much information the presence/absence of a feature contributes to making the correct prediction on the target variable.
            - It can capture non-linear relationships but doesn't account for feature interactions.
            - A score of 0 means the feature and target are independent, while higher scores indicate stronger relationships.
            """)