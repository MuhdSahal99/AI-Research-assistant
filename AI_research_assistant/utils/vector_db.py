import streamlit as st
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from utils.config import load_config
from pinecone import Pinecone, ServerlessSpec  # Updated import

class PineconeService:
    """
    Service for managing document vectors using Pinecone for similarity search.
    """
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.is_connected = False
        self.index = None
        self.dimension = 1024  # Mistral embedding dimension
        self.pc = None  # Pinecone client
        
        # Initialize connection
        self._initialize()
    
    def _initialize(self) -> bool:
        """
        Initialize the Pinecone connection and index.
        
        Returns:
            bool: Success status
        """
        if not self.api_key or not self.environment:
            st.warning("Pinecone API key or environment not configured.")
            return False
            
        try:
            # Initialize Pinecone with new SDK style
            self.pc = Pinecone(api_key=self.api_key)
            
            # Parse environment to get cloud and region
            # Format is typically 'region-cloud', like 'us-east-1-aws'
            parts = self.environment.split('-')
            if len(parts) >= 3:
                # For environments like 'us-east-1-aws'
                cloud = parts[-1]
                region = '-'.join(parts[:-1])
            else:
                # Fallback
                cloud = 'aws'
                region = self.environment
            
            # Check if index exists, create if it doesn't
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                st.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=cloud,
                        region=region
                    )
                )
                time.sleep(1)  # Give time for index to initialize
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            self.is_connected = True
            return True
            
        except Exception as e:
            st.error(f"Error connecting to Pinecone: {str(e)}")
            self.is_connected = False
            return False
    
    def add_document(self, 
                    document_id: str, 
                    vectors: List[List[float]], 
                    metadata: Dict[str, Any],
                    chunk_texts: List[str]) -> bool:
        """
        Add document vectors to the Pinecone index.
        
        Args:
            document_id: Unique ID for the document
            vectors: List of document chunk embeddings
            metadata: Document metadata dictionary
            chunk_texts: List of text chunks corresponding to vectors
            
        Returns:
            bool: Success status
        """
        if not self.is_connected:
            if not self._initialize():
                return False
                
        if len(vectors) == 0 or len(chunk_texts) == 0:
            st.warning(f"No vectors or chunks provided for document {document_id}")
            return False
            
        try:
            # Prepare vector records for upsert
            records = []
            
            for i, (vector, chunk_text) in enumerate(zip(vectors, chunk_texts)):
                # Create a unique ID for each chunk
                chunk_id = f"{document_id}_chunk_{i}"
                
                # Create metadata for this chunk
                chunk_metadata = {
                    **metadata,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "chunk_text": chunk_text[:1000]  # Limit metadata size
                }
                
                # Create the record
                record = {
                    "id": chunk_id,
                    "values": vector,
                    "metadata": chunk_metadata
                }
                
                records.append(record)
            
            # Upsert in batches (Pinecone has a limit)
            batch_size = 100
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                self.index.upsert(vectors=batch)
            
            st.success(f"Added document {document_id} with {len(vectors)} chunks to Pinecone")
            return True
            
        except Exception as e:
            st.error(f"Error adding document to Pinecone: {str(e)}")
            return False
    
    def search_similar(self, 
                      query_vector: List[float], 
                      k: int = 10, 
                      filter: Optional[Dict[str, Any]] = None,
                      include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the Pinecone index.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filter: Optional filter for metadata
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of dictionaries with similarity results
        """
        if not self.is_connected:
            if not self._initialize():
                return []
                
        try:
            # Execute query
            query_results = self.index.query(
                vector=query_vector,
                top_k=k,
                include_metadata=include_metadata,
                filter=filter
            )
            
            # Process results
            results = []
            for match in query_results.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                }
                
                if include_metadata and hasattr(match, 'metadata'):
                    result["metadata"] = match.metadata
                
                results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"Error searching Pinecone index: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all vectors associated with a document from the index.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            bool: Success status
        """
        if not self.is_connected:
            if not self._initialize():
                return False
                
        try:
            # Delete by metadata filter
            self.index.delete(
                filter={"document_id": document_id}
            )
            
            st.success(f"Deleted document {document_id} from Pinecone")
            return True
            
        except Exception as e:
            st.error(f"Error deleting document from Pinecone: {str(e)}")
            return False
    
    def get_document_ids(self) -> List[str]:
        """
        Get list of unique document IDs in the index.
        
        Returns:
            List of document IDs
        """
        if not self.is_connected:
            if not self._initialize():
                return []
                
        try:
            # Use stats to get info about index
            stats = self.index.describe_index_stats()
            
            # This is not directly supported by Pinecone
            # We would need to fetch all vectors and extract unique document IDs
            # This is a placeholder - in a real implementation you'd need
            # to maintain a separate store of document IDs or use a custom approach
            
            st.warning("Retrieving document IDs not directly supported by Pinecone API")
            return []
            
        except Exception as e:
            st.error(f"Error getting document IDs from Pinecone: {str(e)}")
            return []
    
    def get_document_by_id(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks associated with a document ID.
        
        Args:
            document_id: ID of document to retrieve
            
        Returns:
            List of chunks with their vectors and metadata
        """
        if not self.is_connected:
            if not self._initialize():
                return []
                
        try:
            # Query with filter for the document ID
            # Using a dummy vector (will be ignored due to filter)
            query_results = self.index.query(
                vector=[0.0] * self.dimension,
                top_k=1000,  # Set high to get all chunks
                include_metadata=True,
                filter={"document_id": document_id}
            )
            
            # Process results
            chunks = []
            for match in query_results.matches:
                chunk = {
                    "id": match.id,
                    "metadata": match.metadata if hasattr(match, 'metadata') else {}
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            st.error(f"Error getting document from Pinecone: {str(e)}")
            return []

def initialize_vector_db():
    """
    Initialize the vector database service with configuration.
    Returns an instance of PineconeService.
    """
    config = load_config()
    api_key = config.get("PINECONE_API_KEY")
    environment = config.get("PINECONE_ENVIRONMENT")
    index_name = config.get("PINECONE_INDEX_NAME", "ai-research")
    
    if not api_key or not environment:
        st.warning("Pinecone API key or environment not configured. Vector search features will not work.")
    
    return PineconeService(api_key=api_key, environment=environment, index_name=index_name)