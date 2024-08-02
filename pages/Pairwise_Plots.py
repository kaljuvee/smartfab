import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.title("Dana AI - Pairwise Plots")
st.markdown("""
**Steps:**

1. **Upload file** - upload a merged file containing both upstream (explanatory) variables and downstream (target) variables.
2. **Choose the explanatory variable** - choose upstream (explanatory) variables.
3. **Choose the target variable** - choose a downstream (target) variable.
4. **Visualization** - plot the results to visualize pairwise scatter plots and correlation information.
""")

def encode_columns(df):
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    return df, label_encoders

def split_data(df, explanatory_columns):
    explanatory_df = df[explanatory_columns]
    target_df = df.drop(columns=explanatory_columns)
    return explanatory_df, target_df

def get_correlation(data, var, target):
    """
    Calculate the correlation coefficient for two variables in a DataFrame.
    Returns "N/A" if the calculation cannot be performed.
    """
    try:
        x = pd.to_numeric(data[var], errors='coerce').dropna()
        y = pd.to_numeric(data[target], errors='coerce').dropna()
        if len(x) == 0 or len(y) == 0:
            return "N/A (insufficient data)"
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation
    except Exception as e:
        return f"N/A ({str(e)})"
        
def generate_commentary(correlation, var, target):
    if isinstance(correlation, str):
        return f"Could not calculate correlation between {var} and {target}."
    if correlation > 0.7:
        return f"Strong positive correlation between {var} and {target}."
    elif correlation > 0.3:
        return f"Moderate positive correlation between {var} and {target}."
    elif correlation < 0.3 and correlation >= 0:
        return f"Weak positive correlation between {var} and {target}."
    elif correlation < -0.7:
        return f"Strong negative correlation between {var} and {target}."
    elif correlation < -0.3:
        return f"Moderate negative correlation between {var} and {target}."
    elif correlation > -0.3 and correlation < 0:
        return f"Moderate negative correlation between {var} and {target}."
    else:
        return f"Weak or no correlation between {var} and {target}."

# Function to create subplots using Plotly
def create_subplots(data, selected_vars, target):
    n_vars = len(selected_vars)
    n_cols = 2
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=selected_vars)

    for i, var in enumerate(selected_vars):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        fig.add_trace(go.Scatter(x=data[target], y=data[var], mode='markers', name=var), row=row, col=col)

        # Update axis titles for each subplot
        fig.update_xaxes(title_text=target, row=row, col=col)
        fig.update_yaxes(title_text=var, row=row, col=col)

    fig.update_layout(height=300 * n_rows, width=800, title_text=f"Pairwise Plots for {target}")
    return fig

def display_correlation_info(data, selected_vars, target):
    correlation_info = []

    for var in selected_vars:
        correlation = get_correlation(data, var, target)
        commentary = generate_commentary(correlation, var, target)
        correlation_info.append((target, var, correlation, commentary))

    correlation_df = pd.DataFrame(correlation_info, columns=['Target Variable (Downstream)', 'Explanatory Variable (Upstream)', 'Correlation', 'Commentary'])

    correlation_html = correlation_df.to_html(index=False, escape=False)
    correlation_html = correlation_html.replace('<th>', '<th style="font-weight: bold; background-color: #f0f0f0; text-align: left;">')
    st.markdown(correlation_html, unsafe_allow_html=True)

# Function to upload or select existing dataframe
def select_or_upload_dataframe():
    use_default_file = st.checkbox("Use the existing data file")
    default_file_path = "data/load_df.csv"

    if use_default_file:
        df = pd.read_csv(default_file_path)
        st.success("Using the existing data file.")
    else:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("CSV file uploaded successfully!")
        else:
            df = None

    return df

df = select_or_upload_dataframe()

if df is not None:
    st.write("Data Preview:")
    st.write(df)

    # Encode non-numeric columns
    df, label_encoders = encode_columns(df)

    numeric_columns = df.select_dtypes(include=np.number).columns
    explanatory_vars = st.multiselect('Select one or more Explanatory Variables (x):', numeric_columns)
    
    if explanatory_vars:
        explanatory_df, target_df = split_data(df, explanatory_vars)
        selected_target_variable = st.selectbox('Select a target variable (y):', target_df.columns, index=0)

        # Recalculate button
        if st.button('Visualize'):
            fig = create_subplots(df, explanatory_vars, selected_target_variable)
            st.plotly_chart(fig)
            display_correlation_info(df, explanatory_vars, selected_target_variable)
