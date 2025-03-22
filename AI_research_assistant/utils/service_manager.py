import os
import streamlit as st
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import services
from utils.document_processor import DocumentProcessor
from utils.embedding_service import EmbeddingService, initialize_embedding_service
from utils.vector_db import VectorDBService, initialize_vector_db
from utils.llm_service import LLMService, initialize_llm_service

class ServiceManager:
    """
    Manager to coordinate all services for the research platform.
    """
    
    def __init__(self):
        """Initialize the service manager."""
        # Load environment variables
        load_dotenv()
        
        # Initialize state
        self.initialized = False
        self.services_status = {
            "llm": False,
            "embedding": False,
            "vector_db": False
        }
        
        # Initialize services with environment variables
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all services."""
        try:
            # Get API keys from environment
            groq_api_key = os.getenv("GROQ_API_KEY", "")
            mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
            pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
            pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "")
            
            # Initialize document processor
            self.document_processor = DocumentProcessor()
            
            # Initialize LLM service
            if groq_api_key:
                model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
                self.llm_service = initialize_llm_service(groq_api_key, model)
                self.services_status["llm"] = True
            else:
                st.warning("⚠️ GROQ API key missing. Please add it to your .env file.")
                self.llm_service = None
            
            # Initialize embedding service
            if mistral_api_key:
                self.embedding_service = initialize_embedding_service(mistral_api_key)
                self.services_status["embedding"] = True
            else:
                st.warning("⚠️ MISTRAL API key missing. Please add it to your .env file.")
                self.embedding_service = None
            
            # Initialize vector database
            if pinecone_api_key and pinecone_env:
                index_name = os.getenv("PINECONE_INDEX_NAME", "research-papers")
                self.vector_db = initialize_vector_db(pinecone_api_key, pinecone_env)
                self.services_status["vector_db"] = True
            else:
                st.warning("⚠️ PINECONE configuration missing. Please add API key and environment to your .env file.")
                self.vector_db = None
            
            self.initialized = True
            return True
            
        except Exception as e:
            st.error(f"Error initializing services: {str(e)}")
            self.initialized = False
            return False
    
    def process_document(self, uploaded_file) -> Dict[str, Any]:
        """
        Process an uploaded document.
        
        Args:
            uploaded_file: Streamlit uploaded file
            
        Returns:
            Processed document info
        """
        if not self.initialized:
            return {"error": "Services not initialized"}
        
        # Process the document
        return self.document_processor.process_document(uploaded_file)
    
    async def generate_embedding(self, text: str) -> Optional[list]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None
        """
        if not self.initialized or not self.embedding_service:
            return None
        
        return await self.embedding_service.get_embedding(text)
    
    async def store_document_embedding(self, document: Dict[str, Any]) -> bool:
        """
        Generate and store document embedding.
        
        Args:
            document: Processed document
            
        Returns:
            Success status
        """
        if not self.initialized or not self.embedding_service or not self.vector_db:
            return False
        
        try:
            # Generate embedding for the document
            embedding = await self.embedding_service.get_embedding(document["full_text"])
            
            # Store in vector database
            return self.vector_db.store_document(
                document_id=document["document_id"],
                embedding=embedding,
                metadata={
                    "filename": document["filename"],
                    "title": document.get("metadata", {}).get("title", ""),
                    "author": document.get("metadata", {}).get("author", ""),
                    "text_length": document.get("text_length", 0)
                }
            )
        except Exception as e:
            st.error(f"Error storing document embedding: {str(e)}")
            return False
    
    async def analyze_research_quality(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze research quality.
        
        Args:
            document: Processed document
            
        Returns:
            Quality analysis results
        """
        if not self.initialized or not self.llm_service:
            return {"error": "LLM service not initialized"}
        
        return await self.llm_service.analyze_research_quality(document["full_text"])
    
    async def generate_summary(self, document: Dict[str, Any], summary_type: str = "general") -> Dict[str, Any]:
        """
        Generate document summary.
        
        Args:
            document: Processed document
            summary_type: Type of summary
            
        Returns:
            Summary results
        """
        if not self.initialized or not self.llm_service:
            return {"error": "LLM service not initialized"}
        
        return await self.llm_service.generate_summary(document["full_text"], summary_type)
    
    async def check_compliance(self, document: Dict[str, Any], guidelines: str) -> Dict[str, Any]:
        """
        Check document compliance with guidelines.
        
        Args:
            document: Processed document
            guidelines: Compliance guidelines
            
        Returns:
            Compliance check results
        """
        if not self.initialized or not self.llm_service:
            return {"error": "LLM service not initialized"}
        
        return await self.llm_service.check_compliance(document["full_text"], guidelines)
    
    async def compare_documents(self, target_doc: Dict[str, Any], reference_docs: list) -> Dict[str, Any]:
        """
        Compare documents for similarity.
        
        Args:
            target_doc: Target document
            reference_docs: List of reference documents
            
        Returns:
            Similarity analysis results
        """
        if not self.initialized or not self.llm_service:
            return {"error": "Services not initialized"}
        
        # Extract text from reference documents
        reference_texts = [doc["full_text"] for doc in reference_docs]
        
        # Perform semantic similarity using embeddings
        if self.embedding_service:
            target_embedding = await self.embedding_service.get_embedding(target_doc["full_text"])
            similarity_scores = []
            
            for ref_doc in reference_docs:
                ref_embedding = await self.embedding_service.get_embedding(ref_doc["full_text"])
                similarity = EmbeddingService.compute_similarity(target_embedding, ref_embedding)
                similarity_scores.append({
                    "document_id": ref_doc["document_id"],
                    "filename": ref_doc["filename"],
                    "similarity_score": similarity
                })
            
            # Sort by similarity score
            similarity_scores.sort(key=lambda x: x["similarity_score"], reverse=True)
        else:
            similarity_scores = []
        
        # Get detailed LLM analysis
        novelty_analysis = await self.llm_service.detect_similarity(
            target_doc["full_text"], 
            reference_texts
        )
        
        return {
            "similarity_scores": similarity_scores,
            "novelty_analysis": novelty_analysis
        }
    
    def get_services_status(self) -> Dict[str, bool]:
        """
        Get status of all services.
        
        Returns:
            Services status dictionary
        """
        return self.services_status

def initialize_services() -> ServiceManager:
    """
    Initialize all services and return the manager.
    
    Returns:
        Service manager instance
    """
    return ServiceManager()

# Utility function to run async code in Streamlit
def run_async_task(coro):
    """
    Run an async coroutine in Streamlit.
    
    Args:
        coro: Async coroutine
        
    Returns:
        Coroutine result
    """
    return asyncio.run(coro)