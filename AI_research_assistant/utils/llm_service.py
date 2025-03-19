import httpx
import json
import streamlit as st
import time
from typing import Dict, List, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.config import load_config

class GroqLLMService:
    """
    Service for interacting with Groq's Llama3 API for research analysis.
    """
    
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.2, 
        max_tokens: int = 2000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a completion from the LLM using Groq's API.
        
        Args:
            messages: List of message objects with role and content
            temperature: Controls randomness (lower = more deterministic)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the LLM response
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": stream
                }
                
                response = await client.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload
                )
                
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            st.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            st.error(f"Request error occurred: {str(e)}")
            raise
        except Exception as e:
            st.error(f"Error calling Groq API: {str(e)}")
            raise
    
    async def analyze_research_quality(self, document_text: str) -> Dict[str, Any]:
        """
        Analyze the quality of research content using LLM.
        
        Args:
            document_text: The text content of the research document
            
        Returns:
            Dictionary with quality analysis results
        """
        prompt = [
            {"role": "system", "content": 
             """You are a research quality assessment AI. Analyze the provided research text for:
             1. Argument strength and logical coherence
             2. Missing citations or references
             3. Methodology gaps
             4. Structural inconsistencies
             
             Provide a structured analysis with specific examples from the text.
             Format your response as a JSON object with the following structure:
             {
                "overall_quality_score": 1-10,
                "argument_strength": {
                    "score": 1-10,
                    "issues": [{"description": "", "severity": "high/medium/low", "location": ""}]
                },
                "reference_quality": {
                    "score": 1-10,
                    "issues": [{"description": "", "severity": "high/medium/low", "location": ""}]
                },
                "methodology_assessment": {
                    "score": 1-10,
                    "issues": [{"description": "", "severity": "high/medium/low", "location": ""}]
                },
                "structural_consistency": {
                    "score": 1-10,
                    "issues": [{"description": "", "severity": "high/medium/low", "location": ""}]
                },
                "summary": ""
             }"""
            },
            {"role": "user", "content": document_text[:15000]}  # Limit text length
        ]
        
        response = await self.generate_completion(
            messages=prompt,
            temperature=0.1,  # Low temperature for more consistent analysis
            max_tokens=2000
        )
        
        try:
            content = response["choices"][0]["message"]["content"]
            # Extract JSON response - the LLM might wrap the JSON in markdown
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].strip()
            else:
                json_str = content
                
            analysis_result = json.loads(json_str)
            return analysis_result
            
        except Exception as e:
            st.error(f"Error parsing LLM response: {str(e)}")
            return {
                "error": "Failed to parse LLM response",
                "raw_response": response["choices"][0]["message"]["content"]
            }
    
    async def detect_novelty(self, research_text: str, reference_texts: List[str]) -> Dict[str, Any]:
        """
        Detect novel contributions and similarities between research texts.
        
        Args:
            research_text: The text to analyze for novelty
            reference_texts: List of reference texts to compare against
            
        Returns:
            Dictionary with novelty analysis results
        """
        # Combine reference texts with separators (limit length to avoid token limits)
        combined_refs = "\n---\n".join(
            [ref[:3000] for ref in reference_texts[:3]]  # Limit number and size of references
        )
        
        prompt = [
            {"role": "system", "content": 
             """You are a research novelty detection AI. Compare the research document with the reference documents and identify:
             1. Overlapping content and paraphrasing
             2. Novel contributions and findings
             3. Reused sections with minimal changes
             
             Format your response as a JSON object with the following structure:
             {
                "novelty_score": 1-10,
                "overlapping_sections": [
                    {"content": "", "similarity_description": "", "severity": "high/medium/low"}
                ],
                "novel_contributions": [
                    {"description": "", "significance": "high/medium/low"}
                ],
                "recommendation": "",
                "analysis_summary": ""
             }"""
            },
            {"role": "user", "content": 
             f"RESEARCH DOCUMENT:\n\n{research_text[:10000]}\n\nREFERENCE DOCUMENTS:\n\n{combined_refs}"}
        ]
        
        response = await self.generate_completion(
            messages=prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        try:
            content = response["choices"][0]["message"]["content"]
            # Extract JSON response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].strip()
            else:
                json_str = content
                
            novelty_result = json.loads(json_str)
            return novelty_result
            
        except Exception as e:
            st.error(f"Error parsing LLM novelty response: {str(e)}")
            return {
                "error": "Failed to parse LLM response",
                "raw_response": response["choices"][0]["message"]["content"]
            }
    
    async def generate_summary(self, document_text: str, summary_type: str = "general") -> Dict[str, Any]:
        """
        Generate a summary of the research document.
        
        Args:
            document_text: The text content of the research document
            summary_type: Type of summary to generate (general, reviewer, editor)
            
        Returns:
            Dictionary with summary information
        """
        prompt_template = {
            "general": "Create a comprehensive but concise summary of this research paper. Include the main research question, methodology, findings, and contributions.",
            "reviewer": "Create a reviewer-focused summary of this research paper. Highlight key points that a reviewer should focus on, including methodology, claims, evidence, and potential issues.",
            "editor": "Create an editor-focused summary of this research paper. Focus on novelty, significance, technical quality, and potential impact in the field."
        }
        
        prompt = [
            {"role": "system", "content": 
             """You are a research summarization AI. Generate a clear, concise, and informative summary of the provided research document.
             Structure your summary in a way that is easy to read and understand.
             
             Format your response as a JSON object with the following structure:
             {
                "title": "",
                "abstract": "",
                "key_points": ["", "", ""],
                "methodology_summary": "",
                "findings_summary": "",
                "limitations": ["", "", ""],
                "future_work": "",
                "significance": ""
             }"""
            },
            {"role": "user", "content": 
             f"{prompt_template.get(summary_type, prompt_template['general'])}\n\nDOCUMENT:\n\n{document_text[:15000]}"}
        ]
        
        response = await self.generate_completion(
            messages=prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        try:
            content = response["choices"][0]["message"]["content"]
            # Extract JSON response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].strip()
            else:
                json_str = content
                
            summary_result = json.loads(json_str)
            return summary_result
            
        except Exception as e:
            st.error(f"Error parsing LLM summary response: {str(e)}")
            return {
                "error": "Failed to parse LLM response",
                "raw_response": response["choices"][0]["message"]["content"]
            }
    
    async def check_compliance(self, document_text: str, guidelines: str) -> Dict[str, Any]:
        """
        Check if a document complies with specified guidelines.
        
        Args:
            document_text: The text content of the research document
            guidelines: The guidelines to check compliance against
            
        Returns:
            Dictionary with compliance information
        """
        prompt = [
            {"role": "system", "content": 
             """You are a research compliance checker AI. Evaluate the provided document against the specified guidelines.
             Check for formatting, citation style, structural requirements, and other compliance issues.
             
             Format your response as a JSON object with the following structure:
             {
                "overall_compliance_score": 1-10,
                "compliance_issues": [
                    {"section": "", "issue": "", "recommendation": "", "severity": "high/medium/low"}
                ],
                "citation_style_compliance": {
                    "score": 1-10,
                    "issues": [{"description": "", "example": "", "correction": ""}]
                },
                "formatting_compliance": {
                    "score": 1-10,
                    "issues": [{"description": "", "location": "", "correction": ""}]
                },
                "structural_compliance": {
                    "score": 1-10,
                    "issues": [{"description": "", "recommendation": ""}]
                },
                "summary": ""
             }"""
            },
            {"role": "user", "content": 
             f"GUIDELINES:\n\n{guidelines}\n\nDOCUMENT:\n\n{document_text[:15000]}"}
        ]
        
        response = await self.generate_completion(
            messages=prompt,
            temperature=0.1,
            max_tokens=2000
        )
        
        try:
            content = response["choices"][0]["message"]["content"]
            # Extract JSON response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].strip()
            else:
                json_str = content
                
            compliance_result = json.loads(json_str)
            return compliance_result
            
        except Exception as e:
            st.error(f"Error parsing LLM compliance response: {str(e)}")
            return {
                "error": "Failed to parse LLM response",
                "raw_response": response["choices"][0]["message"]["content"]
            }

def initialize_llm_service():
    """
    Initialize the LLM service with configuration.
    Returns an instance of GroqLLMService.
    """
    config = load_config()
    api_key = config.get("GROQ_API_KEY")
    model = config.get("GROQ_MODEL")
    
    if not api_key:
        st.warning("Groq API key not configured. Some features may not work.")
    
    return GroqLLMService(api_key=api_key, model=model)