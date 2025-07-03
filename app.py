import streamlit as st

def main():
    st.set_page_config(
        page_title="Data Clustering App",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("Navigation")
    st.sidebar.info("Select a page to proceed")
    
if __name__ == "__main__":
    main()