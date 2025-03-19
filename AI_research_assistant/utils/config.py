import os
import streamlit as st
from dotenv import load_dotenv

def load_config():
    """
    Load configuration from environment variables only (.env file).
    Returns a dictionary with all configuration values.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Configuration dictionary - only use os.getenv, no st.secrets
    config = {
        # LLM Configuration (Groq)
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
        "GROQ_MODEL": os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        
        # Embedding Configuration (Mistral)
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY", ""),
        "MISTRAL_EMBEDDING_MODEL": os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed"),
        
        # Vector Database Configuration (Pinecone)
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY", ""),
        "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT", ""),
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "research-similarity"),
        
        # Document Processing
        "MAX_DOCUMENT_SIZE_MB": int(os.getenv("MAX_DOCUMENT_SIZE_MB", 15)),
        "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 1000)),
        "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 200)),
        
        # Similarity Configuration
        "SIMILARITY_THRESHOLD": float(os.getenv("SIMILARITY_THRESHOLD", 0.85)),
        
        # Application Settings
        "DEBUG": os.getenv("DEBUG", "False").lower() == "true",
    }
    
    return config

def validate_config(config):
    """
    Validate that required configuration values are present.
    Returns a tuple of (is_valid, missing_keys).
    """
    required_keys = [
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT"
    ]
    
    missing_keys = [key for key in required_keys if not config.get(key)]
    
    return len(missing_keys) == 0, missing_keys

def check_api_keys():
    """
    Check if all required API keys are available.
    Returns a dictionary with the status of each key.
    """
    config = load_config()
    
    return {
        "groq": bool(config.get("GROQ_API_KEY")),
        "mistral": bool(config.get("MISTRAL_API_KEY")),
        "pinecone": bool(config.get("PINECONE_API_KEY")) and bool(config.get("PINECONE_ENVIRONMENT"))
    }