import os
import re
import streamlit as st
import PyPDF2
import docx
import pdfplumber
from typing import List, Dict, Any, Tuple, BinaryIO, Optional
from io import BytesIO
import uuid
from utils.config import load_config

class DocumentProcessor:
    """
    Process uploaded research documents for text extraction and chunking.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.config = load_config()
        
    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Process an uploaded file and extract text content.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dictionary with extracted information
        """
        if uploaded_file is None:
            return {"error": "No file uploaded"}
            
        # Check file size
        if uploaded_file.size > self.config.get("MAX_DOCUMENT_SIZE_MB", 15) * 1024 * 1024:
            return {"error": f"File too large (max {self.config.get('MAX_DOCUMENT_SIZE_MB', 15)}MB)"}
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Get file extension
        filename = uploaded_file.name
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Process file based on extension
        try:
            if file_ext == '.pdf':
                text, metadata = self._extract_pdf_text(uploaded_file)
            elif file_ext == '.docx':
                text, metadata = self._extract_docx_text(uploaded_file)
            elif file_ext == '.txt':
                text, metadata = self._extract_text_file(uploaded_file)
            else:
                return {"error": f"Unsupported file format: {file_ext}"}
                
            # Chunk the text using simple method (no NLTK)
            chunks = self._simple_chunk_text(text)
            
            return {
                "document_id": document_id,
                "filename": filename,
                "file_type": file_ext,
                "full_text": text,
                "chunks": chunks,
                "metadata": metadata,
                "chunk_count": len(chunks)
            }
                
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return {"error": f"Failed to process document: {str(e)}"}
    
    def _extract_pdf_text(self, file: BinaryIO) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            file: File-like object containing PDF data
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        bytes_data = file.getvalue()
        file_stream = BytesIO(bytes_data)
        
        # Try to extract with PyPDF2 first
        try:
            reader = PyPDF2.PdfReader(file_stream)
            metadata = {}
            if reader.metadata:
                for key in reader.metadata:
                    # Clean the metadata key
                    clean_key = key.strip('/').lower()
                    metadata[clean_key] = str(reader.metadata[key])
            
            all_text = []
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                all_text.append(page.extract_text())
                
            text = "\n".join(all_text)
            
            # If PyPDF2 extracts little or no text, try pdfplumber
            if len(text.strip()) < 100:
                file_stream.seek(0)
                return self._extract_with_pdfplumber(file_stream, metadata)
                
            return text, metadata
                
        except Exception as e:
            # Fallback to pdfplumber
            file_stream.seek(0)
            return self._extract_with_pdfplumber(file_stream, {})
    
    def _extract_with_pdfplumber(self, file_stream: BytesIO, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF using pdfplumber (fallback method).
        
        Args:
            file_stream: BytesIO stream of the PDF
            metadata: Existing metadata dictionary
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        with pdfplumber.open(file_stream) as pdf:
            if not metadata and hasattr(pdf, 'metadata') and pdf.metadata:
                metadata = pdf.metadata
                
            all_text = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                all_text.append(page_text)
                
        return "\n".join(all_text), metadata
    
    def _extract_docx_text(self, file: BinaryIO) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from a DOCX file.
        
        Args:
            file: File-like object containing DOCX data
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        bytes_data = file.getvalue()
        file_stream = BytesIO(bytes_data)
        
        doc = docx.Document(file_stream)
        
        # Extract text
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
            
        # Extract metadata
        metadata = {}
        if doc.core_properties:
            props = doc.core_properties
            if props.author:
                metadata['author'] = props.author
            if props.title:
                metadata['title'] = props.title
            if props.created:
                metadata['created'] = str(props.created)
            if props.modified:
                metadata['modified'] = str(props.modified)
            if props.subject:
                metadata['subject'] = props.subject
                
        return "\n".join(full_text), metadata
    
    def _extract_text_file(self, file: BinaryIO) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from a plain text file.
        
        Args:
            file: File-like object containing text data
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        text = file.getvalue().decode('utf-8', errors='ignore')
        
        # Extract basic metadata (just filename for text files)
        metadata = {
            "filename": file.name
        }
        
        return text, metadata
    
    def _simple_chunk_text(self, text: str) -> List[str]:
        """
        Simple method to split text into chunks based on paragraphs and character count.
        Does not rely on NLTK.
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks
        """
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        chunks = []
        
        # First try to split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_length = len(para)
            
            # If adding this paragraph would exceed chunk size, store current chunk and start new one
            if current_length + para_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # For overlap, keep last paragraph
                overlap_text = current_chunk[-1] if len(current_chunk) > 0 else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text) if overlap_text else 0
                
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_length += para_length + 1  # +1 for the space
            
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        # If we couldn't split by paragraphs effectively, use character-based chunking
        if not chunks or len(chunks) == 1 and len(text) > self.chunk_size:
            chunks = []
            start = 0
            
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                
                # Try to end at a period or space
                if end < len(text):
                    # Look for the last period + space or newline
                    last_period = text.rfind('. ', start, end)
                    last_newline = text.rfind('\n', start, end)
                    
                    # Choose the furthest boundary
                    if last_period > start + self.chunk_size // 2:
                        end = last_period + 1
                    elif last_newline > start + self.chunk_size // 2:
                        end = last_newline + 1
                    else:
                        # If no good boundary, at least try to break at a space
                        last_space = text.rfind(' ', start + self.chunk_size // 2, end)
                        if last_space > start:
                            end = last_space
                
                chunks.append(text[start:end].strip())
                
                # Move start position for next chunk, accounting for overlap
                start = end - self.chunk_overlap
                if start < 0 or start >= len(text):
                    break
        
        # Handle very short texts
        if not chunks:
            chunks = [text]
            
        return chunks
    
    def extract_abstract(self, text: str) -> Optional[str]:
        """
        Extract abstract from research paper text.
        
        Args:
            text: Full text of the document
            
        Returns:
            Abstract text or None if not found
        """
        # Common patterns to identify abstract section
        abstract_patterns = [
            r'(?i)abstract[:\s]+([^#]+?)(?:\n\s*\n|$|\n\s*[0-9]+\.|\n\s*keywords|\n\s*introduction)',
            r'(?i)abstract[:\s]+([\s\S]+?)(?:\n\s*\n|$|\n\s*[0-9]+\.|\n\s*keywords|\n\s*introduction)'
        ]
        
        for pattern in abstract_patterns:
            matches = re.search(pattern, text)
            if matches:
                abstract = matches.group(1).strip()
                return abstract
                
        return None
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract main sections from research paper text.
        
        Args:
            text: Full text of the document
            
        Returns:
            Dictionary of section name -> content
        """
        # Common section names in research papers
        section_names = [
            "introduction", "background", "literature review", "related work",
            "methodology", "methods", "experimental setup", "experiments",
            "results", "discussion", "conclusion", "references"
        ]
        
        sections = {}
        
        # Create regex pattern to find sections
        pattern = r'(?i)(?:^|\n\s*)((?:' + '|'.join(section_names) + r')[.\s:]+)([\s\S]+?)(?=\n\s*(?:' + '|'.join(section_names) + r')[.\s:]+|\Z)'
        
        matches = re.finditer(pattern, text)
        for match in matches:
            section_name = match.group(1).strip().lower()
            section_content = match.group(2).strip()
            sections[section_name] = section_content
            
        return sections
    
    def extract_references(self, text: str) -> List[str]:
        """
        Extract references from research paper text.
        
        Args:
            text: Full text of the document
            
        Returns:
            List of reference strings
        """
        # Try to locate references section
        ref_section = None
        
        # First try to find a "References" or "Bibliography" section
        ref_patterns = [
            r'(?i)references[:\s]+([\s\S]+)$',
            r'(?i)bibliography[:\s]+([\s\S]+)$',
            r'(?i)references[:\s]+([\s\S]+?)(?:acknowledgements|appendix)'
        ]
        
        for pattern in ref_patterns:
            matches = re.search(pattern, text)
            if matches:
                ref_section = matches.group(1).strip()
                break
                
        if not ref_section:
            return []
            
        # Now try to split into individual references
        references = []
        
        # Try numbered references first (e.g., [1] Author...)
        numbered_refs = re.findall(r'(?:^|\n)\s*\[\d+\](.*?)(?=(?:^|\n)\s*\[\d+\]|\Z)', ref_section, re.DOTALL)
        if numbered_refs:
            references = [ref.strip() for ref in numbered_refs if ref.strip()]
            
        # If that didn't work, try references by author names (e.g., Smith, J. et al.)
        if not references:
            author_refs = re.split(r'(?:^|\n)\s*(?:[A-Z][a-z]+,\s+[A-Z]\.)', ref_section)
            if len(author_refs) > 1:
                # First item is usually empty or header text
                references = ["Author, " + ref.strip() for ref in author_refs[1:] if ref.strip()]
                
        # If still no references found, just split by newlines and try to clean up
        if not references:
            lines = ref_section.split('\n')
            current_ref = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # If line starts with a number or year in parentheses, probably a new reference
                if re.match(r'^\d+\.|\(\d{4}\)', line):
                    if current_ref:
                        references.append(' '.join(current_ref))
                        current_ref = []
                        
                current_ref.append(line)
                
            # Add the last reference
            if current_ref:
                references.append(' '.join(current_ref))
                
        return references