import streamlit as st
import os
import asyncio
import json
import time
import uuid
from utils.document_processor import DocumentProcessor
from utils.config import load_config
from components.document_uploader import render_document_uploader

# Set page config
st.set_page_config(
    page_title="Summarization",
    page_icon="📝",
    layout="wide"
)

# Load config
config = load_config()

# Page title
st.title("Research Summarization")
st.markdown("Generate concise summaries and reviewer briefs to streamline the review process.")

# Initialize session state
if 'summary_results' not in st.session_state:
    st.session_state.summary_results = None
if 'doc_for_summary' not in st.session_state:
    st.session_state.doc_for_summary = None
if 'in_progress' not in st.session_state:
    st.session_state.in_progress = False

# Get services from session state
llm_service = st.session_state.llm_service if 'llm_service' in st.session_state else None

# Initialize document processor
doc_processor = DocumentProcessor(
    chunk_size=config.get("CHUNK_SIZE", 1000),
    chunk_overlap=config.get("CHUNK_OVERLAP", 200)
)

# Sidebar for document upload and processing
with st.sidebar:
    st.header("Document Upload")
    
    # Upload document for summarization
    uploaded_file = st.file_uploader(
        "Upload Research Document",
        type=["pdf", "docx", "txt"],
        key="doc_uploader_summary"
    )
    
    if uploaded_file:
        if st.button("Process Document for Summarization"):
            with st.spinner("Processing document..."):
                # Process the document
                doc_result = doc_processor.process_uploaded_file(uploaded_file)
                
                if "error" in doc_result:
                    st.error(doc_result["error"])
                else:
                    st.session_state.doc_for_summary = doc_result
                    st.success(f"Processed {doc_result['filename']} ({len(doc_result['full_text'])} characters, {doc_result['chunk_count']} chunks)")
                    
                    # Extract and show abstract if available
                    abstract = doc_processor.extract_abstract(doc_result["full_text"])
                    if abstract:
                        st.markdown("### Abstract")
                        st.markdown(abstract)
    
    # Summarization options
    if 'doc_for_summary' in st.session_state and st.session_state.doc_for_summary:
        st.header("Summarization Options")
        
        summary_type = st.radio(
            "Summary Type",
            options=["General", "Reviewer-Focused", "Editor-Focused"],
            index=0,
            horizontal=True
        )
        
        st.session_state.summary_type = summary_type.lower().replace("-", "_").replace(" ", "_")

# Main content area
if 'doc_for_summary' not in st.session_state or not st.session_state.doc_for_summary:
    st.info("Please upload and process a document for summarization.")
else:
    doc = st.session_state.doc_for_summary
    
    # Document info
    st.markdown(f"## Document: {doc['filename']}")
    
    # Start summarization button
    if st.button("Generate Summary") and not st.session_state.in_progress:
        st.session_state.in_progress = True
        
        with st.spinner("Generating research summary..."):
            try:
                # Function to run async code in a blocking way for Streamlit
                def run_async(coro):
                    return asyncio.run(coro)
                
                # Get summary from LLM
                summary_type = getattr(st.session_state, 'summary_type', 'general')
                summary = run_async(llm_service.generate_summary(
                    doc["full_text"][:15000],  # Limit size for LLM
                    summary_type=summary_type
                ))
                
                # Extract additional document sections
                sections = doc_processor.extract_sections(doc["full_text"])
                references = doc_processor.extract_references(doc["full_text"])
                
                # Store results in session state
                st.session_state.summary_results = {
                    "summary": summary,
                    "summary_type": summary_type,
                    "document_sections": sections,
                    "references": references,
                    "document": doc,
                    "timestamp": time.time()
                }
                
                st.success("Summary generation complete!")
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Error during summarization: {str(e)}")
            finally:
                st.session_state.in_progress = False
    
    # Display summary results if available
    if st.session_state.summary_results:
        results = st.session_state.summary_results
        
        if "summary" in results:
            summary = results["summary"]
            summary_type = results.get("summary_type", "general").replace("_", " ").title()
            
            # Create dashboard layout
            st.header(f"{summary_type} Summary")
            
            # Title and abstract
            if "title" in summary:
                st.subheader(summary["title"])
            
            if "abstract" in summary:
                st.markdown(f"**Abstract:** {summary['abstract']}")
            
            # Create tabs for different sections
            tab1, tab2, tab3 = st.tabs([
                "Key Points", 
                "Methodology & Findings", 
                "Limitations & Significance"
            ])
            
            with tab1:
                if "key_points" in summary:
                    st.subheader("Key Points")
                    
                    for i, point in enumerate(summary["key_points"]):
                        st.markdown(f"**{i+1}.** {point}")
            
            with tab2:
                if "methodology_summary" in summary:
                    st.subheader("Methodology")
                    st.markdown(summary["methodology_summary"])
                
                if "findings_summary" in summary:
                    st.subheader("Findings")
                    st.markdown(summary["findings_summary"])
            
            with tab3:
                if "limitations" in summary:
                    st.subheader("Limitations")
                    
                    for i, limitation in enumerate(summary["limitations"]):
                        st.markdown(f"**{i+1}.** {limitation}")
                
                if "future_work" in summary:
                    st.subheader("Future Work")
                    st.markdown(summary["future_work"])
                
                if "significance" in summary:
                    st.subheader("Significance")
                    st.markdown(summary["significance"])
            
            # Export options
            st.header("Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download as JSON
                results_json = json.dumps(st.session_state.summary_results, default=str, indent=2)
                
                st.download_button(
                    label="Download Summary as JSON",
                    data=results_json,
                    file_name=f"summary_{doc['document_id']}.json",
                    mime="application/json"
                )
            
            with col2:
                # Download as Markdown
                markdown_content = f"# {summary.get('title', 'Research Summary')}\n\n"
                markdown_content += f"## Abstract\n\n{summary.get('abstract', '')}\n\n"
                
                markdown_content += "## Key Points\n\n"
                for point in summary.get('key_points', []):
                    markdown_content += f"* {point}\n"
                markdown_content += "\n"
                
                markdown_content += f"## Methodology\n\n{summary.get('methodology_summary', '')}\n\n"
                markdown_content += f"## Findings\n\n{summary.get('findings_summary', '')}\n\n"
                
                markdown_content += "## Limitations\n\n"
                for limitation in summary.get('limitations', []):
                    markdown_content += f"* {limitation}\n"
                markdown_content += "\n"
                
                markdown_content += f"## Future Work\n\n{summary.get('future_work', '')}\n\n"
                markdown_content += f"## Significance\n\n{summary.get('significance', '')}\n\n"
                
                st.download_button(
                    label="Download Summary as Markdown",
                    data=markdown_content,
                    file_name=f"summary_{doc['document_id']}.md",
                    mime="text/markdown"
                )