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
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.feature_selection import RFE, mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt

# Function to encode non-numeric columns
def encode_columns(df):
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    return df, label_encoders

# Streamlit app title
st.title("Dana AI - Root Cause Analysis")

# Option to use the prebuilt data file
use_default_file = st.checkbox("Use the existing data file.")
default_file_path = "data/load_df.csv"

if use_default_file:
    df = pd.read_csv(default_file_path)
    st.success("Using the existing data file.")
else:
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")
    else:
        df = None

if df is not None:
    st.write("Data Preview:")
    st.write(df)

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
            st.markdown('### 1. Built-in Feature Importance')
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

            st.markdown('### 2. Permutation Importance')
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

            st.markdown('### 3. Mutual Information')
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

            st.markdown('### 4. Recursive Feature Elimination')
            rfe = RFE(estimator=model, n_features_to_select=1)
            rfe.fit(X_train, y_train)
            rfe_importances = pd.DataFrame({'Feature': X_train.columns, 'RFE Ranking': rfe.ranking_}).sort_values('RFE Ranking')
            
            st.subheader("Recursive Feature Elimination (RFE) Ranking")
            st.dataframe(rfe_importances)
            st.markdown("""
            **Interpreting Recursive Feature Elimination (RFE) Ranking:**
            - X-axis: Feature names
            - Y-axis: RFE Ranking (lower is better)
            - Features with lower ranking (shorter bars) are considered more important.
            - RFE recursively removes features and ranks them based on when they were eliminated.
            - Rank 1 indicates the most important feature, with higher ranks being less important.
            - This method considers feature dependencies and can capture complex relationships.
            """)

            # 6. Partial Dependence Plots (PDPs) and Individual Conditional Expectation (ICE) Plots
            st.markdown('### 6. Partial Dependence Plots (PDPs) and Individual Conditional Expectation (ICE) Plots')
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                display = PartialDependenceDisplay.from_estimator(model, X_train, features=[0, 1], ax=ax)  # Example: using the first two features
                st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred while generating PDPs: {e}")
            st.markdown("""
            **Interpreting PDPs and ICE Plots:**
            - X-axis: Feature values
            - Y-axis: Model prediction
            - PDPs show the average effect of a feature on the predicted outcome.
            - ICE plots show the effect of a feature for individual data instances.
            - These plots help to understand feature interactions and the model's behavior for specific feature values.
            """)

            # 7. Correlation Matrix and Pairwise Plots
            st.markdown('### 7. Correlations')

            # Correlation Matrix using Plotly
            corr_matrix = X_train.corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')  # Using a recognized colorscale
            fig.update_layout(title='Correlation Matrix', xaxis_title='Features', yaxis_title='Features')
            st.plotly_chart(fig)
            st.markdown("""
            **Interpreting Correlation Matrix:**
            - The heatmap shows the correlation coefficients between features.
            - Values range from -1 to 1, where 1 indicates a perfect positive correlation, -1 indicates a perfect negative correlation, and 0 indicates no correlation.
            - Helps identify feature groupings and potential multicollinearity issues.
            """)
