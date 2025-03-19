import streamlit as st
from typing import Dict, Any, Callable, Optional
from utils.document_processor import DocumentProcessor

def render_document_uploader(
    key: str,
    on_process: Optional[Callable[[Dict[str, Any]], None]] = None,
    title: str = "Upload Document",
    description: str = "Upload a document for processing",
    allowed_types: list = ["pdf", "docx", "txt"],
    max_size_mb: int = 15,
    show_preview: bool = True
) -> Dict[str, Any]:
    """
    Reusable document uploader component.
    
    Args:
        key: Unique key for the component
        on_process: Callback function when document is processed
        title: Title text
        description: Description text
        allowed_types: List of allowed file extensions
        max_size_mb: Maximum file size in MB
        show_preview: Whether to show document preview
        
    Returns:
        Document processing result dictionary or None if no document processed
    """
    st.markdown(f"### {title}")
    st.markdown(description)
    
    # File uploader
    uploaded_file = st.file_uploader(
        f"Upload Document ({', '.join(allowed_types)})",
        type=allowed_types,
        key=f"file_uploader_{key}"
    )
    
    result = None
    
    if uploaded_file:
        # Check file size
        if uploaded_file.size > max_size_mb * 1024 * 1024:
            st.error(f"File too large (max {max_size_mb}MB)")
            return None
        
        # Process button
        if st.button(f"Process Document", key=f"process_btn_{key}"):
            with st.spinner("Processing document..."):
                # Process the document
                doc_processor = DocumentProcessor()
                result = doc_processor.process_uploaded_file(uploaded_file)
                
                if "error" in result:
                    st.error(result["error"])
                    return None
                else:
                    st.success(f"Processed {result['filename']} ({len(result['full_text'])} characters, {result['chunk_count']} chunks)")
                    
                    # Display document preview if requested
                    if show_preview:
                        with st.expander("Document Preview"):
                            # Show metadata
                            if result["metadata"]:
                                st.subheader("Document Metadata")
                                for key, value in result["metadata"].items():
                                    st.markdown(f"**{key}:** {value}")
                            
                            # Show text preview (first chunk)
                            st.subheader("Text Preview")
                            if result["chunks"]:
                                st.markdown(result["chunks"][0])
                    
                    # Call the callback function if provided
                    if on_process:
                        on_process(result)
    
    return result