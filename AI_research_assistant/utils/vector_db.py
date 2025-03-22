import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_db")

class VectorDBService:
    """
    Simplified Vector Database service using Pinecone.
    """
    
    def __init__(self, api_key: str, environment: str, index_name: str = "research-papers"):
        """
        Initialize the Vector DB service.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., "gcp-starter")
            index_name: Name of the Pinecone index to use
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = 1024  # Default dimension for embeddings
        self.pc = None  # Pinecone client
        self.index = None
        self.is_connected = False
        
        # Try to initialize connection
        if api_key and environment:
            self._initialize()
    
    def _initialize(self) -> bool:
        """Initialize connection to Pinecone."""
        if not self.api_key or not self.environment:
            logger.warning("Pinecone API key or environment not configured")
            return False
            
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                
                # Parse environment string to get cloud and region
                parts = self.environment.split('-')
                if len(parts) >= 3:
                    cloud = parts[-1] 
                    region = '-'.join(parts[:-1])
                else:
                    cloud = 'aws'
                    region = self.environment
                
                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=cloud,
                        region=region
                    )
                )
                # Wait for index to initialize
                time.sleep(2)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            self.is_connected = True
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {str(e)}")
            self.is_connected = False
            return False
    
    def store_document(self, document_id: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """
        Store a document embedding in the vector database.
        
        Args:
            document_id: Unique ID for the document
            embedding: Document embedding vector
            metadata: Document metadata
            
        Returns:
            Success status
        """
        if not self.is_connected:
            if not self._initialize():
                return False
        
        try:
            # Prepare vector record
            record = {
                "id": document_id,
                "values": embedding,
                "metadata": metadata
            }
            
            # Upsert the vector
            self.index.upsert(vectors=[record])
            logger.info(f"Stored document {document_id} in Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document in Pinecone: {str(e)}")
            return False
    
    def find_similar(self, 
                    query_embedding: List[float], 
                    top_k: int = 5, 
                    threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find similar documents based on embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of similar document records
        """
        if not self.is_connected:
            if not self._initialize():
                return []
        
        try:
            # Query the index
            query_result = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Process and filter results
            results = []
            for match in query_result.matches:
                if match.score >= threshold:
                    results.append({
                        "document_id": match.id,
                        "similarity_score": match.score,
                        "metadata": match.metadata
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching for similar documents: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector database.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            Success status
        """
        if not self.is_connected:
            if not self._initialize():
                return False
        
        try:
            # Delete the vector
            self.index.delete(ids=[document_id])
            logger.info(f"Deleted document {document_id} from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document from Pinecone: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with database statistics
        """
        if not self.is_connected:
            if not self._initialize():
                return {"status": "disconnected"}
        
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            return {
                "status": "connected",
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {"status": "error", "message": str(e)}

def initialize_vector_db(api_key: str, environment: str) -> VectorDBService:
    """
    Initialize the vector database service.
    
    Args:
        api_key: Pinecone API key
        environment: Pinecone environment
        
    Returns:
        Vector database service instance
    """
    return VectorDBService(api_key=api_key, environment=environment)