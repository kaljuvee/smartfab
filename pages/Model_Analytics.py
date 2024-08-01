import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, create_model, plot_model, interpret_model, pull
import matplotlib.pyplot as plt
import os

# Streamlit app title
st.title("SmartFAB- Model Analytics")

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
        
        # Pull the setup information into a DataFrame
        setup_df = pull()

        # Display the setup DataFrame
        st.write("Setup Information:")
        st.dataframe(setup_df)

    if st.button("Compare Models"):
        with st.spinner('Comparing models...'):
            # Compare baseline models
            best = compare_models()
            st.session_state['best_model'] = best
            
            # Pull the comparison results into a DataFrame
            comparison_df = pull()
            
            # Display the comparison DataFrame
            st.write("Model Comparison Results:")
            st.dataframe(comparison_df)
            
            # Store model names for dropdown selection
            model_names = ['rf', 'catboost', 'dt', 'et', 'lightgbm']
            st.session_state['model_names'] = model_names

    if st.button("Plot Best Model"):
        if 'best_model' in st.session_state:
            # Plot confusion matrix for the best model
            fig = plot_model(st.session_state['best_model'], plot='confusion_matrix', display_format='streamlit')
            st.pyplot(fig)
        else:
            st.error("Please run 'Compare Models' first to determine the best model.")
    
    if 'model_names' in st.session_state:
        selected_model = st.selectbox("Select a Model to Create", st.session_state['model_names'])

        if st.button("Create Selected Model"):
            model = create_model(selected_model)
            st.session_state['created_model'] = model
            st.success(f"Model {selected_model} created successfully!")

    if st.button("SHAP Explain"):
        if 'created_model' in st.session_state:
            with st.spinner('Generating SHAP summary plot...'):
                # Interpret the selected model using SHAP summary plot
                plot = interpret_model(st.session_state['created_model'], plot='summary', save=True)
                
                # Load the saved plot and display it
                fig = plt.figure(figsize=(10, 8))
                img = plt.imread("SHAP Summary.png")
                plt.imshow(img)
                plt.axis('off')
                st.pyplot(fig)
                plt.close(fig)  # Close the figure to free up memory
                
                if os.path.exists("SHAP Summary.png"):
                    os.remove("SHAP Summary.png")
        else:
            st.error("Please create a model first.")
