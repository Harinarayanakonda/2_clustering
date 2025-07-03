import os
import pandas as pd
from typing import Union
import tempfile

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary directory and return path"""
    try:
        # Create temp directory if not exists
        os.makedirs("temp_uploads", exist_ok=True)
        
        # Save file
        file_path = os.path.join("temp_uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        raise Exception(f"Error saving file: {str(e)}")

def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return os.path.splitext(filename)[1].lower()

def is_valid_file(uploaded_file, selected_type: str) -> bool:
    """Check if uploaded file matches selected type"""
    file_ext = get_file_extension(uploaded_file.name)
    
    type_mapping = {
        "CSV": [".csv"],
        "Excel": [".xlsx", ".xls"],
        "JSON": [".json"],
        "Text": [".txt", ".tsv"]
    }
    
    return file_ext in type_mapping.get(selected_type, [])