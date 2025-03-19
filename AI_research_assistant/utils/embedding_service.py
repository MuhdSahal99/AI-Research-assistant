import httpx
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.config import load_config

class MistralEmbeddingService:
    """
    Service for generating embeddings using Mistral AI's API.
    """
    
    def __init__(self, api_key: str, model: str = "mistral-embed"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.mistral.ai/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.embedding_dimension = 1024  # Mistral embed dimension
        self.max_token_length = 8000  # Maximum tokens for Mistral embedding
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        try:
            # Process in batches to avoid token limits
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Truncate long texts to avoid exceeding token limits
                batch_texts = [self._truncate_text(text) for text in batch_texts]
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    payload = {
                        "model": self.model,
                        "input": batch_texts
                    }
                    
                    response = await client.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # Extract embeddings from response
                    batch_embeddings = [item["embedding"] for item in result["data"]]
                    all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
                
        except httpx.HTTPStatusError as e:
            st.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            st.error(f"Request error occurred: {str(e)}")
            raise
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        # Truncate long text
        text = self._truncate_text(text)
        
        embeddings = await self.get_embeddings([text])
        if embeddings:
            return embeddings[0]
        return [0.0] * self.embedding_dimension  # Return zero vector if embedding fails
    
    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to avoid exceeding the model's token limit.
        
        Args:
            text: Text to truncate
            
        Returns:
            Truncated text
        """
        # Rough approximation: 4 chars per token for English
        char_limit = self.max_token_length * 4
        
        if len(text) <= char_limit:
            return text
        
        # If text is too long, truncate to char_limit
        return text[:char_limit]
    
    async def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get embeddings
        embedding1 = np.array(await self.get_embedding(text1))
        embedding2 = np.array(await self.get_embedding(text2))
        
        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    async def batch_similarities(self, query_text: str, reference_texts: List[str]) -> List[float]:
        """
        Compute similarities between a query text and multiple reference texts.
        
        Args:
            query_text: The query text to compare against references
            reference_texts: List of reference texts
            
        Returns:
            List of similarity scores
        """
        if not reference_texts:
            return []
            
        # Get embeddings
        query_embedding = np.array(await self.get_embedding(query_text))
        reference_embeddings = await self.get_embeddings(reference_texts)
        
        # Compute similarities
        similarities = []
        for ref_embedding in reference_embeddings:
            ref_embedding_array = np.array(ref_embedding)
            
            dot_product = np.dot(query_embedding, ref_embedding_array)
            norm1 = np.linalg.norm(query_embedding)
            norm2 = np.linalg.norm(ref_embedding_array)
            
            if norm1 == 0 or norm2 == 0:
                similarities.append(0.0)
            else:
                similarity = dot_product / (norm1 * norm2)
                similarities.append(float(similarity))
        
        return similarities

def initialize_embedding_service():
    """
    Initialize the embedding service with configuration.
    Returns an instance of MistralEmbeddingService.
    """
    config = load_config()
    api_key = config.get("MISTRAL_API_KEY")
    model = config.get("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
    
    if not api_key:
        st.warning("Mistral API key not configured. Embedding features will not work.")
    
    return MistralEmbeddingService(api_key=api_key, model=model)