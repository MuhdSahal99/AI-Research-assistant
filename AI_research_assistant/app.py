import streamlit as st
import os
from utils.config import load_config, check_api_keys
from utils.llm_service import initialize_llm_service
from utils.embedding_service import initialize_embedding_service
from utils.vector_db import initialize_vector_db
import base64

# Load configuration
config = load_config()

# Set page configuration
st.set_page_config(
    page_title="Research LLM Platform",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "assets", "css", "style.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Initialize services if not already in session state
if 'initialized' not in st.session_state:
    with st.spinner("Initializing services..."):
        # Check if configuration is available
        config = load_config()
        api_status = check_api_keys()
        
        # Initialize services with appropriate warnings
        if not api_status["groq"]:
            st.warning("⚠️ GROQ API key missing. Please add it to your .env file.")
        
        if not api_status["mistral"]:
            st.warning("⚠️ MISTRAL API key missing. Please add it to your .env file.")
            
        if not api_status["pinecone"]:
            st.warning("⚠️ PINECONE configuration missing. Please add API key and environment to your .env file.")
        
        # Initialize services even if keys are missing (will show appropriate warnings in the UI)
        llm_service = initialize_llm_service()
        st.session_state.llm_service = llm_service
        
        embedding_service = initialize_embedding_service()
        st.session_state.embedding_service = embedding_service
        
        vector_db = initialize_vector_db()
        st.session_state.vector_db = vector_db
        
        st.session_state.initialized = True
        st.session_state.api_status = api_status

# Display logo and title
col1, col2 = st.columns([1, 5])
try:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(logo_path):
        col1.image(logo_path, width=100)
except Exception:
    pass

col2.title("Research LLM Platform")
col2.markdown("AI-powered research analysis for similarity detection, quality assessment, summarization, and compliance.")

# Main page content
st.markdown("""
## Welcome to the Research LLM Platform

This platform uses advanced AI to help researchers, publishers, and reviewers improve the quality and integrity of academic research.

### Key Features:

1. **Similarity & Novelty Detection**: Identify overlapping content, paraphrased text, and novel contributions between research papers.

2. **Contextual Quality Assessment**: Evaluate argument strength, logical coherence, and research validity with LLM-powered analysis.

3. **Summarization**: Generate concise summaries and reviewer briefs to streamline the review process.

4. **Compliance Checks**: Ensure papers meet formatting guidelines, citation standards, and other requirements.

### Getting Started:

Use the sidebar to navigate to each feature. You can upload research papers in PDF or DOCX format for analysis.
""")

# Display system status
st.sidebar.header("System Status")
if st.session_state.initialized:
    st.sidebar.success("Services initialized")
    
    # Get API status
    api_status = st.session_state.api_status if 'api_status' in st.session_state else check_api_keys()
    
    # LLM Status
    if api_status["groq"]:
        st.sidebar.markdown("**LLM Service:** ✅ Connected (Groq)")
    else:
        st.sidebar.markdown("**LLM Service:** ❌ Not configured (Groq)")
    
    # Embedding Status
    if api_status["mistral"]:
        st.sidebar.markdown("**Embedding Service:** ✅ Connected (Mistral)")
    else:
        st.sidebar.markdown("**Embedding Service:** ❌ Not configured (Mistral)")
    
    # Vector DB Status
    if api_status["pinecone"] and hasattr(st.session_state.vector_db, 'is_connected') and st.session_state.vector_db.is_connected:
        st.sidebar.markdown("**Vector Database:** ✅ Connected (Pinecone)")
    else:
        st.sidebar.markdown("**Vector Database:** ❌ Not configured (Pinecone)")
    
    # Show configuration instructions if needed
    if not all(api_status.values()):
        with st.sidebar.expander("📝 Configuration Instructions"):
            st.markdown("""
            To configure API keys, create a `.env` file in the project root with:
            ```
            GROQ_API_KEY=your_groq_api_key
            MISTRAL_API_KEY=your_mistral_api_key
            PINECONE_API_KEY=your_pinecone_api_key
            PINECONE_ENVIRONMENT=your_pinecone_environment
            ```
            """)
else:
    st.sidebar.error("Services not initialized")

# Footer
st.markdown("---")
st.markdown("© 2025 Research LLM Platform")