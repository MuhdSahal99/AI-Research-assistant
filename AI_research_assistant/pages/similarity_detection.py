import streamlit as st
import os
import asyncio
import json
import time
import uuid
import sys
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.document_processor import DocumentProcessor
from utils.config import load_config
from components.document_uploader import render_document_uploader
from components.similarity_viewer import render_similarity_heatmap, render_similarity_network

# Set page config
st.set_page_config(
    page_title="Similarity & Novelty Detection",
    page_icon="🔍",
    layout="wide"
)

# Load config
config = load_config()

# Page title
st.title("Research Similarity & Novelty Detection")
st.markdown("Upload research papers to identify overlaps, paraphrased content, and novel contributions.")

# Initialize session state
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'uploaded_doc_id' not in st.session_state:
    st.session_state.uploaded_doc_id = None
if 'reference_doc_ids' not in st.session_state:
    st.session_state.reference_doc_ids = []
if 'in_progress' not in st.session_state:
    st.session_state.in_progress = False
if 'primary_doc' not in st.session_state:
    st.session_state.primary_doc = None
if 'reference_docs' not in st.session_state:
    st.session_state.reference_docs = []

# Get services from session state
llm_service = st.session_state.llm_service if 'llm_service' in st.session_state else None
embedding_service = st.session_state.embedding_service if 'embedding_service' in st.session_state else None
vector_db = st.session_state.vector_db if 'vector_db' in st.session_state else None

# Initialize document processor (no chunking)
doc_processor = DocumentProcessor()

# Sidebar for document management
with st.sidebar:
    st.header("Document Management")
    
    # Upload primary document
    primary_tab, reference_tab = st.tabs(["Primary Document", "Reference Documents"])
    
    with primary_tab:
        st.markdown("Upload the primary research document to analyze:")
        primary_file = st.file_uploader(
            "Upload Primary Document",
            type=["pdf", "docx", "txt"],
            key="primary_doc_uploader"
        )
        
        if primary_file:
            if st.button("Process Primary Document"):
                with st.spinner("Processing document..."):
                    # Process the document
                    doc_result = doc_processor.process_uploaded_file(primary_file)
                    
                    if "error" in doc_result:
                        st.error(doc_result["error"])
                    else:
                        st.session_state.primary_doc = doc_result
                        st.session_state.uploaded_doc_id = doc_result["document_id"]
                        st.success(f"Processed {doc_result['filename']} ({len(doc_result['full_text'])} characters)")
                        
                        # Extract and show abstract if available
                        abstract = doc_processor.extract_abstract(doc_result["full_text"]) if hasattr(doc_processor, 'extract_abstract') else None
                        if abstract:
                            st.markdown("### Abstract")
                            st.markdown(abstract)
    
    with reference_tab:
        st.markdown("Upload reference documents to compare against:")
        reference_file = st.file_uploader(
            "Upload Reference Document",
            type=["pdf", "docx", "txt"],
            key="reference_doc_uploader",
            accept_multiple_files=True
        )
        
        if reference_file:
            if st.button("Process Reference Documents"):
                with st.spinner("Processing reference documents..."):
                    reference_docs = []
                    for ref_file in reference_file:
                        # Process the document
                        ref_result = doc_processor.process_uploaded_file(ref_file)
                        
                        if "error" in ref_result:
                            st.error(f"Error processing {ref_file.name}: {ref_result['error']}")
                        else:
                            reference_docs.append(ref_result)
                            st.success(f"Processed {ref_result['filename']}")
                    
                    if reference_docs:
                        st.session_state.reference_docs = reference_docs
                        st.session_state.reference_doc_ids = [doc["document_id"] for doc in reference_docs]

# Main content area
main_col1, main_col2 = st.columns([3, 2])

with main_col1:
    st.header("Document Comparison")
    
    if 'primary_doc' not in st.session_state or not st.session_state.primary_doc:
        st.info("Please upload and process a primary document to analyze.")
    elif not st.session_state.reference_docs:
        st.info("Please upload and process reference documents to compare against.")
    else:
        st.markdown(f"Comparing **{st.session_state.primary_doc['filename']}** against {len(st.session_state.reference_docs)} reference documents.")
        
        # Display control options
        similarity_threshold = st.slider(
            "Similarity Threshold", 
            min_value=0.5, 
            max_value=1.0, 
            value=config.get("SIMILARITY_THRESHOLD", 0.85),
            step=0.05,
            help="Minimum similarity score to consider content as similar"
        )
        
        # Start analysis button
        if st.button("Start Similarity Analysis") and not st.session_state.in_progress:
            st.session_state.in_progress = True
            
            with st.spinner("Analyzing document similarity..."):
                try:
                    # Function to run async code in a blocking way for Streamlit
                    def run_async(coro):
                        return asyncio.run(coro)
                    
                    primary_doc = st.session_state.primary_doc
                    reference_docs = st.session_state.reference_docs
                    
                    # Get embedding for primary document (whole document)
                    primary_text = primary_doc["full_text"]
                    primary_embedding = run_async(embedding_service.get_embedding(primary_text))
                    
                    # Store the embedding in Pinecone (optional)
                    if vector_db and vector_db.is_connected:
                        vector_db.add_document(
                            document_id=primary_doc["document_id"],
                            vectors=[primary_embedding],  # Single vector for the whole document
                            metadata=primary_doc["metadata"],
                            chunk_texts=[primary_text]  # Single chunk = whole text
                        )
                    
                    # Collect reference documents text
                    reference_texts = []
                    reference_metadata = []
                    similarity_results = []
                    
                    # Process each reference document
                    for ref_doc in reference_docs:
                        ref_text = ref_doc["full_text"]
                        reference_texts.append(ref_text)
                        reference_metadata.append({
                            "document_id": ref_doc["document_id"],
                            "filename": ref_doc["filename"],
                            "metadata": ref_doc["metadata"]
                        })
                        
                        # Get embedding for this reference document
                        ref_embedding = run_async(embedding_service.get_embedding(ref_text))
                        
                        # Compute similarity
                        similarity = np.dot(primary_embedding, ref_embedding) / (
                            np.linalg.norm(primary_embedding) * np.linalg.norm(ref_embedding)
                        )
                        
                        # If similarity is above threshold, add to results
                        if similarity >= similarity_threshold:
                            similarity_results.append({
                                "reference_doc_id": ref_doc["document_id"],
                                "reference_filename": ref_doc["filename"],
                                "similarity_score": float(similarity)
                            })
                    
                    # Get LLM novelty analysis
                    novelty_analysis = run_async(llm_service.detect_novelty(
                        primary_doc["full_text"][:15000],  # Limit size for LLM
                        [text[:5000] for text in reference_texts[:3]]  # Limit number and size of references
                    ))
                    
                    # Sort by similarity score
                    similarity_results.sort(key=lambda x: x["similarity_score"], reverse=True)
                    
                    # Store results in session state
                    st.session_state.comparison_results = {
                        "novelty_analysis": novelty_analysis,
                        "similarity_results": similarity_results,
                        "primary_doc": primary_doc,
                        "reference_docs": reference_docs,
                        "timestamp": time.time()
                    }
                    
                    st.success("Analysis complete!")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                finally:
                    st.session_state.in_progress = False