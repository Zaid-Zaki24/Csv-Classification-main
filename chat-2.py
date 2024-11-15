import importlib.util
from config import model
import autopep8
import os
import pandas as pd
import streamlit as st
import chardet
from utils import extract_csv_content, save_uploaded_file

def detect_file_encoding(file_path):
    """
    Detect the encoding of a file using chardet.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Detected encoding or None if not detected.
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result.get("encoding")


def read_csv_with_fallback(file_path):
    """
    Attempt to read a CSV file with a detected encoding, then fallback to common encodings.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    # Step 1: Detect encoding
    detected_encoding = detect_file_encoding(file_path)
    st.info(f"Detected encoding: {detected_encoding}")

    # Step 2: Try detected encoding
    encodings_to_try = [detected_encoding, 'ISO-8859-1', 'Windows-1252', 'utf-8']
    for encoding in encodings_to_try:
        if encoding is None:
            continue
        try:
            st.info(f"Trying encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding)
            st.success(f"File read successfully with encoding: {encoding}")
            return df
        except Exception as e:
            st.warning(f"Encoding '{encoding}' failed: {e}")

    # Step 3: All attempts failed
    st.error("All attempts to read the file failed. Please check the file format.")
    return None


def summarize_csv(file_path):
    """
    Summarize the CSV file by returning the number of columns,
    their names, and unique values for each column.

    Args:
        file_path (str): The path to the uploaded CSV file.

    Returns:
        dict: Summary of the CSV file or an error message.
    """
    df = read_csv_with_fallback(file_path)
    if df is None:
        return {"error": "Failed to read the file with all attempted encodings."}

    # Prepare the summary
    summary = {
            "total_columns": len(df.columns),
            "column_details": {}
        }

    for column in df.columns:
            unique_values = df[column].dropna().unique()
            summary["column_details"][column] = {
                "unique_values": unique_values[:5],  # Limit to 5 unique values for brevity
                "data_type": df[column].dtype.name  # Include data type information
            }

    return summary, df


def save_uploaded_file(uploaded_file, save_directory):
    """
    Saves the uploaded file to the specified directory and processes it.

    Args:
        uploaded_file (streamlit.uploadedfile.UploadedFile): The uploaded file.
        save_directory (str): The directory where the file will be saved.

    Returns:
        str: The path where the file is saved.
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save the uploaded file
        file_name = "file.csv"  # Customize the file name if needed
        file_path = os.path.join(save_directory, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None




import time
import matplotlib.pyplot as plt
import google.generativeai as genai
genai.configure(api_key="AIzaSyAcd2N4aYzFkOQNHF_dRt0u_fbwV6rXUVI")
def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file
def plot_graph_with_gemini(file_path, summary):
    """
    Upload the file to Gemini, extract code for visualization, and render the graphs dynamically.
    
    Args:
        file_path (str): Path to the CSV file.
        summary (str): Summary of the dataset to provide context for the query.
    """
    import re

    try:
        # Upload the file to Gemini
        files = [upload_to_gemini(file_path, mime_type="text/csv")]

        # Wait for files to be processed
        wait_for_files_active(files)

        # Start chat session with Gemini
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        files[0],
                    ],
                },
            ]
        )

        # Query for data visualization
        query = "Can you visualize the data trends?"
        # response = chat_session.send_message(
        #     f"""Query: {query}
        #     For above query give me the code for that and here is summary of the dataset of which user has asked the query. 
        #     The CSV file contains data on issue classifications. Here's a summary of each column, including its name and unique values.
        #     Note: The counting mentioned in the summary may not be accurate, so code should read the file from the local path.Dont use try and catch and read the data using file path and dont give installing requirements.
        #     Code will be given as ```Code here``` in this section nothing else than code as i will extract the code and execute it as it is.
        #     Dont plot the graph only make a function with name plot_graph annd return graph through this fucntion.
        #     Summary:
        #     {summary}
        #     """
        # )

        response = chat_session.send_message(
                f"""Query: {query}
                For the above query, provide the code for the requested functionality. Here is the summary of the dataset the user has asked about.
                The CSV file contains data on issue classifications. Here's a summary of each column, including its name and unique values.
                Note: The counting mentioned in the summary may not be accurate, so the code should read the file from the local path. 
                Don't use try and catch, and read the data using the file path directly without additional error handling. 
                The code must be in Streamlit format and should include widgets for interactivity (e.g., file uploader, dropdowns). 
                Do not include requirements installation or unnecessary comments. 

                Code must:
                - Be encapsulated in a function named `plot_graph` with argument file_path to take csv.
                - Display plots in streamlit.
                - Use the provided `summary` for any assumptions.
                - Use the markdown syntax to enclose the code block like this: ```Code here```.
                - Donot excute this code just make function.

                Summary:
                {summary}
                """
            )

        
        file_path = r"data/file.csv"

        # Extract code blocks from the response
        code_blocks = re.findall(r"```(?:\w+\n)?(.*?)```", response.text, re.DOTALL)
        code_blocks = [block.strip() for block in code_blocks]
        code_to_execute = "\n".join(code_blocks)
        
        
        # Replace `file_path` with the actual file path
        modified_code = re.sub(
            r".*?\.csv.*",
            f"file_path = r'{file_path}'",  # Use raw string in replacement
            code_to_execute
        )
        code_to_execute = modified_code

        formatted_code = autopep8.fix_code(code_to_execute)

        # Save the corrected code for debugging purposes
        with open("temp_script.py", "w") as f:
            f.write(formatted_code)
        script_path = "temp_script.py"
        spec = importlib.util.spec_from_file_location("temp_script", script_path)
        temp_script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_script)

        # Execute the plot_graph function dynamically
        if hasattr(temp_script, "plot_graph"):
            graph_visual = temp_script.plot_graph(file_path)
        else:
            print("Function plot_graph does not exist in temp_script.py.")
            

    

    except Exception as e:
        st.error(f"An error occurred: {e}")


def main():
    """
    Main function for the Streamlit app.
    """
    # Streamlit page configuration
    st.set_page_config(page_title="Chat and Visualize Your CSV", layout="wide")

    # Sidebar for file upload
    st.sidebar.title("Upload Your File")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    # Save directory for uploaded files
    save_directory = "data"
    os.makedirs(save_directory, exist_ok=True)

    # Initialize session state
    if "saved_file_path" not in st.session_state:
        st.session_state.saved_file_path = None
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None

    # Process uploaded file
    if uploaded_file is not None:
        saved_file_path = save_uploaded_file(uploaded_file, save_directory)
        st.session_state.saved_file_path = saved_file_path

        # Generate summary
        summary, df = summarize_csv(saved_file_path)
        st.session_state.dataframe = df
        
        if "error" in summary:
            st.error(f"Error in processing the file: {summary['error']}")
        else:
            st.write("### File Summary")
            # st.write(f"**Total Columns:** {summary['total_columns']}")
            for column, details in summary["column_details"].items():
                st.write(f"- **Column Name:** {column}")
                st.write(f"  - **Data Type:** {details['data_type']}")
                st.write(f"  - **Unique Values:** {', '.join(map(str, details['unique_values']))}")
    
    # Tabs for Chat and Data Visualization
    tab1, tab2 = st.tabs(["Chat with CSV", "Data Visualization"])

    # Tab 1: Chat with CSV
    with tab1:
        st.title("Chat with Your CSV")
        st.write("Ask questions about the uploaded CSV file.")
        
        # Chat input
        if st.session_state.saved_file_path:
            chat_container = st.container()
            
            with chat_container:
                user_question = st.text_input("Your Question:", key="user_input")
                if user_question:
                    # Get response from Gemini
                    #st.write(f"**You:** {user_question}")
                    chat = model.start_chat(
                        history=[
                            {"role": "user", "parts": extract_csv_content(st.session_state.saved_file_path)},
                            {"role": "user", "parts": user_question}
                        ]
                    )
                    response = chat.send_message("PLz provide me answers from above csv file.")
                    
                    #st.write(f"**Gemini:** {response}")
                   
                    
                    
                    response_text = getattr(response, 'text', 'No response content available.')
                    st.write(f"**Response of the content:** {response_text}")

                    
                   

    # Tab 2: Data Visualization
    with tab2:
        st.title("Data Visualization with Gemini")

        if st.session_state.saved_file_path is not None:
            file_path = st.session_state.saved_file_path
            
            # Generate dataset summary
            summary, df = summarize_csv(file_path)
            if "error" in summary:
                st.error(f"Error in generating summary: {summary['error']}")
            else:
                # Display dataset summary
                st.write("### Dataset Summary")
                st.write(f"**Total Columns:** {summary['total_columns']}")
                for column, details in summary["column_details"].items():
                    st.write(f"- **Column Name:** {column}")
                    st.write(f"  - **Data Type:** {details['data_type']}")
                    st.write(f"  - **Unique Values:** {', '.join(map(str, details['unique_values']))}")

                # Trigger Gemini-based visualization
                if st.button("Generate Visualization"):
                    st.write("### Visualization")
                    figure = plot_graph_with_gemini(file_path, summary)
                    # st.pyplot(fig = figure)
        else:
            st.info("Please upload a CSV file to visualize the data.")

if __name__ == "__main__":
    main()
