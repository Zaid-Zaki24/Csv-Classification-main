# import os
# import time
# import google.generativeai as genai
# import streamlit as st
# import pandas as pd

# # Configure the Gemini API
# genai.configure(api_key="AIzaSyAcd2N4aYzFkOQNHF_dRt0u_fbwV6rXUVI")

# # Upload to Gemini function
# def upload_to_gemini(path, mime_type=None):
#     """Uploads the given file to Gemini."""
#     file = genai.upload_file(path, mime_type=mime_type)
#     print(f"Uploaded file '{file.display_name}' as: {file.uri}")
#     return file

# # Wait for files to be active
# def wait_for_files_active(files):
#     """Waits for the given files to be active."""
#     print("Waiting for file processing...")
#     for name in (file.name for file in files):
#         file = genai.get_file(name)
#         while file.state.name == "PROCESSING":
#             time.sleep(10)
#             file = genai.get_file(name)
#         if file.state.name != "ACTIVE":
#             raise Exception(f"File {file.name} failed to process")
#     st.write("...all files ready")

# # Initialize Gemini model
# generation_config = {
#     "temperature": 1,
#     "top_p": 0.95,
#     "top_k": 40,
#     "max_output_tokens": 8192,
#     "response_mime_type": "text/plain",
# }
# model = genai.GenerativeModel(
#     model_name="gemini-1.5-flash",
#     generation_config=generation_config,
# )

# # Streamlit UI
# st.title("Gemini Chat with CSV Support")
# st.write("Upload a CSV file to analyze and ask questions!")

# # File upload and chat session initialization
# if 'chat_session' not in st.session_state:
#     st.session_state.chat_session = None
#     st.session_state.gemini_file = None
#     st.session_state.temp_path = None

# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# if uploaded_file:
#     # Save the uploaded file to a temporary location
#     temp_path = f"temp_{uploaded_file.name}"
#     with open(temp_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Upload the file to Gemini
#     with st.spinner("Uploading to Gemini..."):
#         gemini_file = upload_to_gemini(temp_path, mime_type="text/csv")
#         wait_for_files_active([gemini_file])

#     # Start a new chat session
#     st.session_state.gemini_file = gemini_file
#     st.session_state.chat_session = model.start_chat(
#         history=[
#             {
#                 "role": "user",
#                 "parts": [gemini_file],
#             },
#         ]
#     )
#     st.session_state.temp_path = temp_path
#     st.success("File uploaded and chat session started! Ask your question below:")

# # Chat functionality
# if st.session_state.chat_session:
#     st.write("### Chat with Gemini:")
#     user_input = st.text_input("Your Question:", key="user_question")
#     if user_input:
#         with st.spinner("Getting response from Gemini..."):
#             response = st.session_state.chat_session.send_message(user_input)
#             st.write("### Gemini Response:")
#             st.write(response.text)

# # Clear session when a new file is uploaded
# if uploaded_file and st.session_state.temp_path != f"temp_{uploaded_file.name}":
#     st.session_state.chat_session = None
#     st.session_state.gemini_file = None
#     st.session_state.temp_path = None


# import streamlit as st
# import pandas as pd
# import openai
# import matplotlib.pyplot as plt
# import io

# # Set your OpenAI API key
# openai.api_key = "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"

# # Helper function to summarize CSV data
# def summarize_csv(file):
#     df = pd.read_csv(file)
#     num_rows, num_cols = df.shape
#     columns = df.columns.tolist()
#     summary = f"The uploaded CSV file has {num_rows} rows and {num_cols} columns. The columns are: {', '.join(columns)}."
#     return summary, df

# # Helper function to query GPT for Python code
# def query_gpt_for_code(prompt):
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are a Python expert specialized in data visualization."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         return response['choices'][0]['message']['content']
#     except Exception as e:
#         return f"Error: {e}"

# # Helper function to execute generated code
# def execute_code(code, df):
#     try:
#         # Prepare the environment for code execution
#         local_env = {'df': df, 'plt': plt, 'io': io}
#         exec(code, {}, local_env)
#         buffer = io.BytesIO()
#         plt.savefig(buffer, format="png")
#         plt.close()
#         buffer.seek(0)
#         return buffer
#     except Exception as e:
#         return f"Error in code execution: {e}"

# # Streamlit App
# st.title("AI-Powered CSV Graph Generator")

# # Upload CSV file
# uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# if uploaded_file:
#     st.subheader("File Summary")
#     summary, df = summarize_csv(uploaded_file)
#     st.write(summary)

#     # Display first few rows
#     st.dataframe(df.head())

#     # Get user query
#     user_query = st.text_area("Describe the graph you want to generate (e.g., 'Show a bar chart of column X').")

#     if user_query:
#         # Generate Python code for the graph
#         st.subheader("Generated Code")
#         code_prompt = (
#             f"The user uploaded a CSV file. {summary} "
#             f"Write Python code using matplotlib to create the following graph based on the data: {user_query}"
#         )
#         generated_code = query_gpt_for_code(code_prompt)
#         st.code(generated_code)

#         # Execute the code and display the graph
#         if st.button("Generate Graph"):
#             result = execute_code(generated_code, df)
#             if isinstance(result, io.BytesIO):
#                 st.image(result, caption="Generated Graph")
#             else:
#                 st.error(result)


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

class DataVisualizationAgent:
    def __init__(self):
        """Initialize the agent."""
        self._setup_plotting_style()
        
    def _setup_plotting_style(self) -> None:
        """Set up the plotting style."""
        try:
            sns.set_style("whitegrid")
            sns.set_palette("husl")
        except Exception as e:
            st.warning(f"Could not set plotting style: {str(e)}")

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get numeric columns from DataFrame."""
        return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    def _get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Get categorical columns from DataFrame."""
        return df.select_dtypes(include=['object', 'category']).columns.tolist()

    def _create_visualization(self, df: pd.DataFrame, query: str) -> None:
        """Create visualizations based on user query."""
        fig = plt.figure(figsize=(15, 10))
        
        if any(word in query.lower() for word in ['trend', 'time', 'over time', 'timeline']):
            self._plot_time_series(df, fig)
        elif any(word in query.lower() for word in ['distribution', 'spread', 'range']):
            self._plot_distribution(df, fig)
        elif any(word in query.lower() for word in ['compare', 'comparison', 'versus', 'vs']):
            self._plot_comparison(df, fig)
        elif any(word in query.lower() for word in ['category', 'categories', 'group', 'groups']):
            self._plot_categories(df, fig)
        elif any(word in query.lower() for word in ['correlation', 'relationship', 'between']):
            self._plot_correlation(df, fig)
        else:
            self._plot_summary(df, fig)
        
        st.pyplot(fig)

    def _plot_time_series(self, df: pd.DataFrame, fig: plt.Figure) -> None:
        """Plot time series."""
        numeric_cols = self._get_numeric_columns(df)
        for i, col in enumerate(numeric_cols[:4]):
            ax = fig.add_subplot(2, 2, i + 1)
            df[col].plot(ax=ax)
            ax.set_title(f'{col} Over Time')
            ax.tick_params(axis='x', rotation=45)

    def _plot_distribution(self, df: pd.DataFrame, fig: plt.Figure) -> None:
        """Plot distributions."""
        numeric_cols = self._get_numeric_columns(df)
        for i, col in enumerate(numeric_cols[:4]):
            ax = fig.add_subplot(2, 2, i + 1)
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.tick_params(axis='x', rotation=45)

    def _plot_comparison(self, df: pd.DataFrame, fig: plt.Figure) -> None:
        """Plot comparisons."""
        numeric_cols = self._get_numeric_columns(df)
        categorical_cols = self._get_categorical_columns(df)
        if numeric_cols and categorical_cols:
            ax = fig.add_subplot(1, 1, 1)
            sns.boxplot(data=df, x=categorical_cols[0], y=numeric_cols[0], ax=ax)
            ax.set_title(f'{numeric_cols[0]} by {categorical_cols[0]}')
            ax.tick_params(axis='x', rotation=45)

    def _plot_categories(self, df: pd.DataFrame, fig: plt.Figure) -> None:
        """Plot category counts."""
        categorical_cols = self._get_categorical_columns(df)
        for i, col in enumerate(categorical_cols[:4]):
            ax = fig.add_subplot(2, 2, i + 1)
            df[col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f'Count by {col}')
            ax.tick_params(axis='x', rotation=45)

    def _plot_correlation(self, df: pd.DataFrame, fig: plt.Figure) -> None:
        """Plot correlation heatmap."""
        numeric_cols = self._get_numeric_columns(df)
        if len(numeric_cols) > 1:
            ax = fig.add_subplot(1, 1, 1)
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap')

    def _plot_summary(self, df: pd.DataFrame, fig: plt.Figure) -> None:
        """Plot summary visualizations."""
        numeric_cols = self._get_numeric_columns(df)
        if numeric_cols:
            ax = fig.add_subplot(2, 2, 1)
            df[numeric_cols].boxplot(ax=ax)
            ax.set_title('Numeric Columns Summary')
            ax.tick_params(axis='x', rotation=45)


# Streamlit app
def main():
    st.title("Data Visualization Agent")
    st.write("Upload a CSV file and query your data for visualizations.")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Data")
        st.write(df.head())

        query = st.text_input("Enter your query (e.g., 'Show trends over time')")
        if query:
            agent = DataVisualizationAgent()
            st.write(f"### Visualization for: {query}")
            agent._create_visualization(df, query)

if __name__ == "__main__":
    main()
