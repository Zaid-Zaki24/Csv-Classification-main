import csv
import os
import time

import pandas as pd
import streamlit as st
import chardet

def extract_csv_content(pathname: str) -> list[str]:
    """
    Extracts the content of a CSV file and returns it as a list of strings.

    Args:
        pathname (str): The path to the CSV file.

    Returns:
        list[str]: A list containing the content of the CSV file, with start and end indicators.
    """
    parts = [f"--- START OF CSV {pathname} ---"]
    with open(pathname, "r", newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            parts.append(" ".join(row))
    parts.append(f"--- END OF CSV {pathname} ---")
    return parts


def detect_file_encoding(file_path):
    """
    Detects the encoding of a file using chardet.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The detected encoding or 'utf-8' as default.
    """
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
        return result.get("encoding", "utf-8")  # Default to 'utf-8' if no encoding detected


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

        # Detect the file encoding
        encoding = detect_file_encoding(file_path)

        # Create a DataFrame from the saved CSV file
        try:
            df = pd.read_csv(file_path, encoding=encoding)
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return None

        # Display success message and preview the DataFrame
        success_message = st.success(f"File saved successfully with encoding: {encoding}")
        time.sleep(1)  # Wait for 1 second (optional)
        success_message.empty()  # Clear the success message

        st.write("### Preview of the DataFrame")
        st.write(df.head())  # Display the first few rows

        return file_path
    except Exception as e:
        st.error(f"An error occurred while saving the file: {e}")
        return None