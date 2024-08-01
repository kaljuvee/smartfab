import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, plot_model

# Streamlit app title
st.title("Model Analytics")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # Select target variable
    target_variable = st.selectbox("Select the target variable", df.columns)

    if st.button("Setup Model"):
        # Initialize PyCaret setup
        s = setup(data=df, target=target_variable, session_id=123)
        st.success("Model setup completed!")

    if st.button("Compare Models"):
        # Compare baseline models
        best = compare_models()
        st.session_state['best_model'] = best
        st.write("Best Model:")
        st.write(best)

    if st.button("Plot Model"):
        if 'best_model' in st.session_state:
            # Plot confusion matrix for the best model
            plot_model(st.session_state['best_model'], plot='confusion_matrix')
            st.pyplot()
        else:
            st.error("Please run 'Compare Models' first to determine the best model.")
