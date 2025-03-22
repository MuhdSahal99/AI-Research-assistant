import httpx
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_service")

class LLMService:
    """
    Service for interacting with Groq LLM API for research tasks.
    """
    
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        """
        Initialize the LLM service.
        
        Args:
            api_key: Groq API key
            model: LLM model to use
        """
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
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Generate completion using the LLM.
        
        Args:
            messages: List of message objects (system, user, assistant)
            temperature: Controls randomness
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                response = await client.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload
                )
                
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise
    
    async def analyze_research_quality(self, document_text: str) -> Dict[str, Any]:
        """
        Analyze research quality using the LLM.
        
        Args:
            document_text: Research document text
            
        Returns:
            Quality analysis results
        """
        # Truncate text to a reasonable size
        text = document_text[:5000]
        
        prompt = [
            {"role": "system", "content": 
             """You are a research quality assessment AI. Analyze the provided research text for:
             1. Argument strength and logical coherence
             2. Missing citations or references
             3. Methodology gaps
             4. Structural inconsistencies
             
             Format your response as a JSON object with the following structure:
             {
                "overall_quality_score": 1-10,
                "argument_strength": {
                    "score": 1-10,
                    "issues": [{"description": "", "severity": "high/medium/low"}]
                },
                "reference_quality": {
                    "score": 1-10,
                    "issues": [{"description": "", "severity": "high/medium/low"}]
                },
                "methodology_assessment": {
                    "score": 1-10,
                    "issues": [{"description": "", "severity": "high/medium/low"}]
                },
                "structural_consistency": {
                    "score": 1-10,
                    "issues": [{"description": "", "severity": "high/medium/low"}]
                },
                "summary": ""
             }"""
            },
            {"role": "user", "content": text}
        ]
        
        response = await self.generate_completion(
            messages=prompt,
            temperature=0.1,
            max_tokens=2000
        )
        
        try:
            content = response["choices"][0]["message"]["content"]
            
            # Look for start of JSON after introductory text
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "Could not find valid JSON in response", "content": content}
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {"error": "Failed to parse response", "raw_response": str(response)}
    
    async def detect_similarity(self, target_text: str, reference_texts: List[str]) -> Dict[str, Any]:
        """
        Detect similarities between research papers.
        
        Args:
            target_text: Target research paper text
            reference_texts: List of reference paper texts
            
        Returns:
            Similarity analysis results
        """
        # Truncate texts
        target = target_text[:10000]
        refs = [ref[:3000] for ref in reference_texts[:3]]
        
        combined_refs = "\n---\n".join(refs)
        
        prompt = [
            {"role": "system", "content": 
             """You are a research similarity detection AI. Compare the target document with the reference documents and identify:
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
             f"TARGET DOCUMENT:\n\n{target}\n\nREFERENCE DOCUMENTS:\n\n{combined_refs}"}
        ]
        
        response = await self.generate_completion(
            messages=prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        try:
            content = response["choices"][0]["message"]["content"]
            # Extract JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].strip()
            else:
                json_str = content
                
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"Error parsing similarity response: {str(e)}")
            return {
                "error": "Failed to parse response",
                "raw_response": response["choices"][0]["message"]["content"]
            }
    
    async def generate_summary(self, document_text: str, summary_type: str = "general") -> Dict[str, Any]:
        """
        Generate research paper summary.
        
        Args:
            document_text: Research document text
            summary_type: Type of summary (general, reviewer, editor)
            
        Returns:
            Summary results
        """
        # Truncate text
        text = document_text[:15000]
        
        # Different prompt based on summary type
        type_prompts = {
            "general": "Create a comprehensive but concise summary of this research paper.",
            "reviewer_focused": "Create a reviewer-focused summary highlighting methodology, claims, and potential issues.",
            "editor_focused": "Create an editor-focused summary focusing on novelty, significance, and impact."
        }
        
        type_prompt = type_prompts.get(summary_type, type_prompts["general"])
        
        prompt = [
            {"role": "system", "content": 
             f"""You are a research summarization AI. {type_prompt}
             
             Format your response as a JSON object with the following structure:
             {{
                "title": "",
                "abstract": "",
                "key_points": ["", "", ""],
                "methodology_summary": "",
                "findings_summary": "",
                "limitations": ["", "", ""],
                "future_work": "",
                "significance": ""
             }}"""
            },
            {"role": "user", "content": text}
        ]
        
        response = await self.generate_completion(
            messages=prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        try:
            content = response["choices"][0]["message"]["content"]
            # Extract JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].strip()
            else:
                json_str = content
                
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"Error parsing summary response: {str(e)}")
            return {
                "error": "Failed to parse response",
                "raw_response": response["choices"][0]["message"]["content"]
            }
    
    async def check_compliance(self, document_text: str, guidelines: str) -> Dict[str, Any]:
        """
        Check document compliance with guidelines.
        
        Args:
            document_text: Research document text
            guidelines: Compliance guidelines text
            
        Returns:
            Compliance check results
        """
        # Truncate text
        text = document_text[:3000]
        
        prompt = [
            {"role": "system", "content": 
             """You are a research compliance checker AI. Evaluate the document against the specified guidelines.
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
             f"GUIDELINES:\n\n{guidelines}\n\nDOCUMENT:\n\n{text}"}
        ]
        
        response = await self.generate_completion(
            messages=prompt,
            temperature=0.1,
            max_tokens=2000
        )
        
        try:
            content = response["choices"][0]["message"]["content"]
            
            # Extract only the JSON portion
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return {"error": "JSON parsing failed", "content": content}
            else:
                return {"error": "No JSON found in response", "content": content}
                
        except Exception as e:
            logger.error(f"Error parsing compliance response: {str(e)}")
            return {"error": "Failed to parse response", "raw_response": str(response)}

def initialize_llm_service(api_key: str, model: str = "llama3-8b-8192") -> LLMService:
    """
    Initialize the LLM service.
    
    Args:
        api_key: Groq API key
        model: LLM model to use
        
    Returns:
        LLM service instance
    """
    return LLMService(api_key=api_key, model=model)