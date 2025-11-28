"""
Gemini AI helper module for the LLM Analysis Quiz application.

This module provides integration with Google's Gemini API for:
- Text analysis and question answering
- Audio transcription and understanding (native multimodal support)
- Code generation for data analysis
- Image analysis (if needed)

Design choice: Gemini 2.0 Flash is used as default for its excellent
multimodal capabilities (text, audio, images) and fast response times.
"""

import logging
import json
import re
from typing import Optional, Any, Dict, List
import pandas as pd

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class GeminiHelper:
    """
    Helper class for Gemini AI operations.
    
    Provides methods for text completion, audio transcription,
    and data analysis code generation using Google's Gemini API.
    """
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.settings = get_settings()
        self.client = None
        self.model = None
        
        if not GEMINI_AVAILABLE:
            logger.warning("google-generativeai package not installed")
            return
            
        if not self.settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not configured")
            return
        
        try:
            genai.configure(api_key=self.settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=self.settings.GEMINI_MODEL,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            self.client = genai
            logger.info(f"Gemini initialized with model: {self.settings.GEMINI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if Gemini is available and configured."""
        return self.model is not None
    
    async def transcribe_audio(self, audio_data: bytes, mime_type: str = "audio/opus") -> Optional[str]:
        """
        Transcribe audio using Gemini's native audio understanding.
        
        Gemini can directly process audio files without a separate transcription API.
        
        Args:
            audio_data: Raw audio file bytes.
            mime_type: MIME type of the audio (e.g., "audio/opus", "audio/mp3", "audio/wav").
            
        Returns:
            Transcribed text, or None if transcription fails.
        """
        if not self.is_available:
            logger.warning("Gemini not available for audio transcription")
            return None
        
        try:
            # Upload the audio file
            logger.info(f"Transcribing audio ({len(audio_data)} bytes, {mime_type})")
            
            # Create inline data for the audio
            audio_part = {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": audio_data
                }
            }
            
            # Ask Gemini to transcribe
            prompt = """Listen to this audio carefully and transcribe exactly what is said.
If there are numbers, write them as digits (e.g., "123" not "one hundred twenty three").
If there are instructions or questions, include them exactly.
Provide ONLY the transcription, nothing else."""
            
            response = self.model.generate_content([prompt, audio_part])
            
            if response and response.text:
                transcription = response.text.strip()
                logger.info(f"Audio transcription: {transcription[:200]}...")
                return transcription
            
            logger.warning("Empty response from Gemini audio transcription")
            return None
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return None
    
    async def analyze_with_audio(
        self, 
        question_text: str, 
        audio_data: bytes, 
        mime_type: str = "audio/opus",
        additional_context: str = ""
    ) -> Optional[str]:
        """
        Analyze a question that includes audio content.
        
        Args:
            question_text: The text question/context.
            audio_data: Raw audio file bytes.
            mime_type: MIME type of the audio.
            additional_context: Additional context (e.g., CSV data summary).
            
        Returns:
            The answer as a string.
        """
        if not self.is_available:
            return None
        
        try:
            audio_part = {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": audio_data
                }
            }
            
            prompt = f"""You are solving a data analysis quiz. 

Question/Context from the page:
{question_text}

{f"Additional data context: {additional_context}" if additional_context else ""}

Listen to the audio carefully - it may contain:
- The actual question to answer
- Numbers or values to use in calculations
- Instructions on what operation to perform

Based on the audio and context, determine and provide ONLY the final answer.
If the answer is a number, provide just the number.
If it's text, provide just the text.
No explanations, just the answer."""

            response = self.model.generate_content([prompt, audio_part])
            
            if response and response.text:
                answer = response.text.strip()
                logger.info(f"Audio analysis answer: {answer}")
                return answer
            
            return None
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return None
    
    async def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> Optional[str]:
        """
        Get a chat completion from Gemini.
        
        Args:
            system_prompt: System instructions.
            user_prompt: User message.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in response.
            
        Returns:
            Response text, or None if failed.
        """
        if not self.is_available:
            return None
        
        try:
            # Gemini doesn't have separate system prompts, so combine them
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            if response and response.text:
                return response.text.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Gemini chat completion failed: {e}")
            return None
    
    async def generate_analysis_code(
        self,
        question: str,
        df: pd.DataFrame,
        additional_context: str = ""
    ) -> Optional[str]:
        """
        Generate Python code to analyze a DataFrame and answer a question.
        
        Args:
            question: The question to answer.
            df: The DataFrame to analyze.
            additional_context: Additional context (e.g., from audio transcription).
            
        Returns:
            Python code string that sets 'answer' variable.
        """
        if not self.is_available:
            return None
        
        # Create DataFrame summary
        df_info = f"""DataFrame Info:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Dtypes: {df.dtypes.to_dict()}
- First 5 rows:
{df.head().to_string()}
- Statistics:
{df.describe().to_string()}"""
        
        prompt = f"""You are a Python data analyst. Generate code to answer this question.

Question: {question}
{f"Additional context: {additional_context}" if additional_context else ""}

{df_info}

Requirements:
1. The DataFrame is already loaded as 'df'
2. pandas is imported as 'pd', numpy as 'np'
3. Your code MUST set a variable called 'answer' with the final result
4. Return ONLY executable Python code, no markdown or explanations
5. The answer should be a single value (number, string, or simple list)

Important patterns:
- "below cutoff X" or "less than X" means: df[df[col] < X][col].sum()
- "above cutoff X" means: df[df[col] > X][col].sum()
- "count where" means: len(df[condition])

Generate the Python code:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 500}
            )
            
            if response and response.text:
                code = response.text.strip()
                
                # Clean up code - remove markdown if present
                code = re.sub(r'^```python\s*', '', code)
                code = re.sub(r'^```\s*', '', code)
                code = re.sub(r'\s*```$', '', code)
                code = code.strip()
                
                logger.info(f"Generated analysis code:\n{code}")
                return code
            
            return None
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return None
    
    async def interpret_question(
        self,
        question_text: str,
        available_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Interpret a quiz question to understand what's being asked.
        
        Args:
            question_text: The question text.
            available_data: Dictionary of available data sources.
            
        Returns:
            Interpretation dict with 'task', 'data_source', 'parameters'.
        """
        if not self.is_available:
            return None
        
        prompt = f"""Analyze this quiz question and determine what's being asked.

Question: {question_text}

Available data sources: {list(available_data.keys())}

Respond with a JSON object containing:
- "task": what operation to perform (e.g., "sum", "count", "filter", "extract")
- "data_source": which data source to use
- "parameters": any parameters mentioned (e.g., cutoff values, filters)
- "answer_type": expected answer type ("number", "text", "list")

JSON response:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 300}
            )
            
            if response and response.text:
                text = response.text.strip()
                # Extract JSON from response
                json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            
            return None
            
        except Exception as e:
            logger.error(f"Question interpretation failed: {e}")
            return None
    
    async def extract_answer(self, text: str) -> Optional[str]:
        """
        Extract the final answer from a response text.
        
        Args:
            text: Text that may contain an answer.
            
        Returns:
            Extracted answer.
        """
        if not self.is_available:
            return None
        
        prompt = f"""Extract ONLY the final answer from this text. 
If it's a number, return just the number.
If it's text, return just the relevant text.
No explanations.

Text: {text}

Answer:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0, "max_output_tokens": 100}
            )
            
            if response and response.text:
                return response.text.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Answer extraction failed: {e}")
            return None

    async def analyze_image(self, image_data: bytes, mime_type: str = "image/png") -> Optional[str]:
        """
        Analyze an image and describe its contents.
        
        Args:
            image_data: Raw image file bytes.
            mime_type: MIME type (e.g., "image/png", "image/jpeg", "image/webp").
            
        Returns:
            Description of the image contents.
        """
        if not self.is_available:
            return None
        
        try:
            logger.info(f"Analyzing image ({len(image_data)} bytes, {mime_type})")
            
            image_part = {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            }
            
            prompt = """Analyze this image and describe what you see.
If it contains text, transcribe it exactly.
If it contains data (charts, tables, numbers), extract the values.
If it contains a question or instructions, include them.
Be precise and detailed."""
            
            response = self.model.generate_content([prompt, image_part])
            
            if response and response.text:
                result = response.text.strip()
                logger.info(f"Image analysis: {result[:200]}...")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return None

    async def extract_text_from_image(self, image_data: bytes, mime_type: str = "image/png") -> Optional[str]:
        """
        Extract text from an image (OCR).
        
        Args:
            image_data: Raw image file bytes.
            mime_type: MIME type of the image.
            
        Returns:
            Extracted text from the image.
        """
        if not self.is_available:
            return None
        
        try:
            image_part = {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            }
            
            prompt = """Extract ALL text visible in this image.
Transcribe exactly what you see, preserving:
- Numbers (as digits)
- Special characters
- Line breaks where appropriate
Return ONLY the extracted text, nothing else."""
            
            response = self.model.generate_content([prompt, image_part])
            
            if response and response.text:
                return response.text.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Text extraction from image failed: {e}")
            return None

    async def analyze_with_image(
        self,
        question_text: str,
        image_data: bytes,
        mime_type: str = "image/png",
        additional_context: str = ""
    ) -> Optional[str]:
        """
        Answer a question based on image content.
        
        Args:
            question_text: The question/context.
            image_data: Raw image file bytes.
            mime_type: MIME type of the image.
            additional_context: Additional context (e.g., data).
            
        Returns:
            The answer as a string.
        """
        if not self.is_available:
            return None
        
        try:
            image_part = {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            }
            
            prompt = f"""You are solving a data analysis quiz.

Question/Context: {question_text}
{f"Additional context: {additional_context}" if additional_context else ""}

Look at the image carefully - it may contain:
- Data tables or charts
- Instructions or questions
- Numbers or values needed for calculations

Based on the image and context, provide ONLY the final answer.
If the answer is a number, provide just the number.
If it's text, provide just the text.
No explanations."""

            response = self.model.generate_content([prompt, image_part])
            
            if response and response.text:
                answer = response.text.strip()
                logger.info(f"Image analysis answer: {answer}")
                return answer
            
            return None
            
        except Exception as e:
            logger.error(f"Image analysis with question failed: {e}")
            return None


# Convenience function
def get_gemini_helper() -> GeminiHelper:
    """Get a GeminiHelper instance."""
    return GeminiHelper()
