import httpx
import numpy as np
import logging
from typing import List, Dict, Any, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding_service")

class EmbeddingService:
    """
    Service for generating text embeddings using Mistral AI API.
    """
    
    def __init__(self, api_key: str, model: str = "mistral-embed"):
        """
        Initialize the embedding service.
        
        Args:
            api_key: Mistral API key
            model: Embedding model to use
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.mistral.ai/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.max_token_length = 8000  # Maximum tokens for embedding
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Truncate text if needed
        text = self._truncate_text(text)
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": self.model,
                    "input": [text]
                }
                
                response = await client.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract embedding
                if "data" in result and len(result["data"]) > 0:
                    return result["data"][0]["embedding"]
                else:
                    logger.error("No embedding data returned from API")
                    return []
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in an efficient batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Truncate texts if needed
        truncated_texts = [self._truncate_text(text) for text in texts]
        
        try:
            # Process in batches of 10 to avoid API limits
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(truncated_texts), batch_size):
                batch = truncated_texts[i:i+batch_size]
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    payload = {
                        "model": self.model,
                        "input": batch
                    }
                    
                    response = await client.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # Extract embeddings
                    batch_embeddings = [item["embedding"] for item in result["data"]]
                    all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in batch embedding: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error in batch embedding: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to avoid exceeding token limits.
        
        Args:
            text: Text to truncate
            
        Returns:
            Truncated text
        """
        # Rough approximation: 4 chars per token for English
        char_limit = self.max_token_length * 4
        
        if len(text) <= char_limit:
            return text
        
        return text[:char_limit]
    
    @staticmethod
    def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1)
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

def initialize_embedding_service(api_key: str) -> EmbeddingService:
    """
    Initialize the embedding service.
    
    Args:
        api_key: Mistral API key
        
    Returns:
        Embedding service instance
    """
    return EmbeddingService(api_key=api_key)