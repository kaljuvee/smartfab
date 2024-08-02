import streamlit as st
import pandas as pd
import os
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define model
model = 'gpt-4o'

# Page configuration
st.set_page_config(page_title="Dana")
st.title("Dana GPT")

# Function to read the DataFrame
@st.cache_data
def read_df(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

# Option to use the prebuilt data file
use_default_file = st.checkbox("Use the existing data file.")
default_file_path = "data/load_df.csv"

df = None

if use_default_file:
    if os.path.exists(default_file_path):
        df = read_df(default_file_path)
        st.success("Using the existing data file.")
    else:
        st.error(f"The file {default_file_path} does not exist.")
else:
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = read_df(uploaded_file)
        if df is not None:
            st.success("CSV file uploaded successfully!")

if df is not None:
    st.write("Data Preview:")
    st.write(df)

    # Function to process a question
    def process_question(question):
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        llm = ChatOpenAI(
            temperature=0, model=model, openai_api_key=openai_api_key, streaming=True
        )

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,  # Set verbose to False to avoid detailed execution messages
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            allow_dangerous_code=True  # Allow dangerous code execution
        )

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            try:
                response = pandas_df_agent.run(question, callbacks=[st_cb])
                # Filter out the unwanted line
                filtered_response = "\n".join(line for line in response.split("\n") if not line.startswith("python_repl_ast:"))
            except Exception as e:
                filtered_response = str(e)

            st.session_state.messages.append({"role": "assistant", "content": filtered_response})
            st.write(filtered_response)

    # Initialize or clear conversation history
    if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Display conversation history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Sample questions
    sample_questions = [
        "How many unique values are there in the MODEL column?", 
        "How many rows are there total in this data set?", 
        "Give me the standard deviation, min and max values in the column BILANCELLE/ASSALE",
        "How many unique values are there in the column TEAM_LOAD?",
        "What is the total count of TRUE values in the column BIL_PREVIOUS_EMPTY?"
    ]

    # Create columns for sample questions
    num_columns = 3
    num_questions = len(sample_questions)
    num_rows = (num_questions + num_columns - 1) // num_columns
    columns = st.columns(num_columns)

    # Add buttons for sample questions
    for i in range(num_questions):
        col_index = i % num_columns
        row_index = i // num_columns

        with columns[col_index]:
            if columns[col_index].button(sample_questions[i]):
                process_question(sample_questions[i])

    # User input for new questions
    container = st.container()
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Ask a question:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            process_question(user_input)
else:
    st.info("Please upload a CSV file to continue.")
