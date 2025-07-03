import streamlit as st
import pandas as pd
import os
from utils.data_processing import save_uploaded_file, get_file_extension, is_valid_file
from utils.clustering import detect_data_type

# Page configuration
st.set_page_config(page_title="Data Upload", layout="wide")
st.title("📊 Data Upload and Preview")

# Create temp directory if not exists
os.makedirs("temp_uploads", exist_ok=True)

def show_data_info(df):
    """Display dataset information and sample records"""
    st.success("File uploaded successfully!")
    
    # Display dataset dimensions
    cols = st.columns(2)
    cols[0].metric("Number of Rows", df.shape[0])
    cols[1].metric("Number of Columns", df.shape[1])
    
    # Display sample records
    st.subheader("Sample Records (First 5 rows)")
    st.dataframe(df.head(), use_container_width=True)
    
    # Store the dataframe in session state
    st.session_state['uploaded_df'] = df
    st.session_state['data_type'] = detect_data_type(df)
    
    # Enable navigation to next page
    st.session_state['data_uploaded'] = True

def main():
    # File upload section
    with st.expander("📤 Upload Dataset", expanded=True):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            file_type = st.selectbox(
                "Select File Format",
                ["CSV", "Excel", "JSON", "Text"],
                help="Select the format of your data file"
            )
        
        with col2:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["csv", "xlsx", "xls", "json", "txt"],
                help="Upload your dataset file"
            )
    
    if uploaded_file is not None:
        # Validate file
        if not is_valid_file(uploaded_file, file_type):
            st.error(f"Uploaded file doesn't match selected type: {file_type}")
            return
        
        # Save the file temporarily
        file_path = save_uploaded_file(uploaded_file)
        
        try:
            # Read file based on type
            if file_type == "CSV":
                df = pd.read_csv(file_path)
            elif file_type == "Excel":
                df = pd.read_excel(file_path)
            elif file_type == "JSON":
                df = pd.read_json(file_path)
            elif file_type == "Text":
                df = pd.read_csv(file_path, delimiter='\t')
            
            show_data_info(df)
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Navigation button (only enabled after successful upload)
    if st.session_state.get('data_uploaded', False):
        st.divider()
        st.page_link("pages/2_Clustering.py", label="Go to Clustering Analysis →", icon="➡️")

if __name__ == "__main__":
    if 'data_uploaded' not in st.session_state:
        st.session_state['data_uploaded'] = False
    main()