"""
LLM Helper module for AI integration.

This module provides a unified interface for LLM operations:
- Primary: Google Gemini (supports audio, images, text natively)
- Fallback: OpenAI via AI Pipe or direct API

The module automatically selects the best available provider based on
configuration and the type of content being processed.

Design Choice: Provider Auto-Selection
--------------------------------------
- If GEMINI_API_KEY is set and content includes audio/images -> Use Gemini
- If only OPENAI_API_KEY is set -> Use OpenAI
- LLM_PROVIDER setting can force a specific provider
"""

import json
import logging
import base64
from typing import Optional, Dict, Any, List, Union
import pandas as pd

from openai import AsyncOpenAI, OpenAI

from app.core.config import get_settings
from app.services.gemini_helper import GeminiHelper, GEMINI_AVAILABLE

logger = logging.getLogger(__name__)


class LLMHelper:
    """
    Unified helper class for LLM operations.
    
    Automatically selects between Gemini and OpenAI based on:
    - Available API keys
    - Content type (audio requires Gemini)
    - User preference (LLM_PROVIDER setting)
    """
    
    def __init__(self):
        """Initialize LLM helper with available providers."""
        self.settings = get_settings()
        self._openai_client: Optional[AsyncOpenAI] = None
        self._openai_sync_client: Optional[OpenAI] = None
        self._gemini_helper: Optional[GeminiHelper] = None
        
        # Initialize Gemini if available
        if self.settings.GEMINI_API_KEY and GEMINI_AVAILABLE:
            self._gemini_helper = GeminiHelper()
            if self._gemini_helper.is_available:
                logger.info("Gemini provider initialized")
            else:
                self._gemini_helper = None
                
    @property
    def gemini(self) -> Optional[GeminiHelper]:
        """Get Gemini helper if available."""
        return self._gemini_helper
    
    @property
    def openai_client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if not self._openai_client:
            if not self.settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not configured")
            self._openai_client = AsyncOpenAI(
                api_key=self.settings.OPENAI_API_KEY,
                base_url=self.settings.OPENAI_BASE_URL
            )
        return self._openai_client
    
    @property
    def openai_sync_client(self) -> OpenAI:
        """Get or create sync OpenAI client."""
        if not self._openai_sync_client:
            if not self.settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not configured")
            self._openai_sync_client = OpenAI(
                api_key=self.settings.OPENAI_API_KEY,
                base_url=self.settings.OPENAI_BASE_URL
            )
        return self._openai_sync_client
    
    def is_available(self) -> bool:
        """Check if any LLM provider is available."""
        has_gemini = self._gemini_helper is not None and self._gemini_helper.is_available
        has_openai = bool(self.settings.OPENAI_API_KEY)
        return has_gemini or has_openai
    
    def _should_use_gemini(self, has_audio: bool = False, has_image: bool = False) -> bool:
        """Determine if Gemini should be used for this request."""
        provider = self.settings.LLM_PROVIDER.lower()
        
        # Force specific provider
        if provider == "gemini":
            return self._gemini_helper is not None and self._gemini_helper.is_available
        if provider == "openai":
            return False
        
        # Auto mode: prefer Gemini for multimodal, otherwise use what's available
        if has_audio or has_image:
            # Multimodal content requires Gemini
            return self._gemini_helper is not None and self._gemini_helper.is_available
        
        # For text-only, prefer Gemini if available (it's free tier friendly)
        if self._gemini_helper is not None and self._gemini_helper.is_available:
            return True
        
        return False
    
    async def transcribe_audio(self, audio_data: bytes, mime_type: str = "audio/opus") -> Optional[str]:
        """
        Transcribe audio content.
        
        Args:
            audio_data: Raw audio bytes.
            mime_type: Audio MIME type.
            
        Returns:
            Transcribed text or None.
        """
        if self._gemini_helper and self._gemini_helper.is_available:
            return await self._gemini_helper.transcribe_audio(audio_data, mime_type)
        
        logger.warning("Audio transcription requires Gemini - not available")
        return None
    
    async def analyze_with_audio(
        self,
        question_text: str,
        audio_data: bytes,
        mime_type: str = "audio/opus",
        additional_context: str = ""
    ) -> Optional[str]:
        """
        Analyze a question with audio content using Gemini.
        
        Args:
            question_text: Text context/question.
            audio_data: Raw audio bytes.
            mime_type: Audio MIME type.
            additional_context: Additional context.
            
        Returns:
            Answer or None.
        """
        if self._gemini_helper and self._gemini_helper.is_available:
            return await self._gemini_helper.analyze_with_audio(
                question_text, audio_data, mime_type, additional_context
            )
        
        logger.warning("Audio analysis requires Gemini - not available")
        return None

    async def analyze_image(self, image_data: bytes, mime_type: str = "image/png") -> Optional[str]:
        """
        Analyze an image and describe its contents.
        
        Args:
            image_data: Raw image bytes.
            mime_type: Image MIME type.
            
        Returns:
            Description or None.
        """
        if self._gemini_helper and self._gemini_helper.is_available:
            return await self._gemini_helper.analyze_image(image_data, mime_type)
        
        logger.warning("Image analysis requires Gemini - not available")
        return None

    async def extract_text_from_image(self, image_data: bytes, mime_type: str = "image/png") -> Optional[str]:
        """
        Extract text from an image (OCR).
        
        Args:
            image_data: Raw image bytes.
            mime_type: Image MIME type.
            
        Returns:
            Extracted text or None.
        """
        if self._gemini_helper and self._gemini_helper.is_available:
            return await self._gemini_helper.extract_text_from_image(image_data, mime_type)
        
        logger.warning("Image OCR requires Gemini - not available")
        return None

    async def analyze_with_image(
        self,
        question_text: str,
        image_data: bytes,
        mime_type: str = "image/png",
        additional_context: str = ""
    ) -> Optional[str]:
        """
        Analyze a question with image content using Gemini.
        
        Args:
            question_text: Text context/question.
            image_data: Raw image bytes.
            mime_type: Image MIME type.
            additional_context: Additional context.
            
        Returns:
            Answer or None.
        """
        if self._gemini_helper and self._gemini_helper.is_available:
            return await self._gemini_helper.analyze_with_image(
                question_text, image_data, mime_type, additional_context
            )
        
        logger.warning("Image analysis requires Gemini - not available")
        return None
    
    async def interpret_question(self, question_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM to interpret a quiz question and extract requirements.
        
        Tries Gemini first, falls back to OpenAI.
        """
        # Try Gemini first
        if self._should_use_gemini():
            result = await self._gemini_helper.interpret_question(
                question_text, 
                {"context": context} if context else {}
            )
            if result:
                return result
        
        # Fall back to OpenAI
        return await self._interpret_question_openai(question_text, context)
    
    async def _call_openai_with_fallback(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 1000,
        json_mode: bool = False
    ) -> str:
        """
        Call OpenAI with fallback to secondary model if primary fails.
        """
        models_to_try = [self.settings.OPENAI_MODEL]
        if self.settings.OPENAI_FALLBACK_MODEL:
            models_to_try.append(self.settings.OPENAI_FALLBACK_MODEL)
            
        last_error = None
        
        for model in models_to_try:
            try:
                logger.info(f"Attempting LLM call with model: {model}")
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                    
                response = await self.openai_client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
                
            except Exception as e:
                logger.warning(f"LLM call failed with model {model}: {e}")
                last_error = e
                continue
                
        raise last_error or Exception("All LLM models failed")

    async def _interpret_question_openai(self, question_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """OpenAI implementation of interpret_question."""
        system_prompt = """You are an expert data analyst assistant. Your job is to interpret quiz questions about data analysis and extract the specific requirements.

Analyze the question and return a JSON object with:
{
    "task_type": "one of: calculation, filtering, aggregation, visualization, transformation, statistics, ml_analysis",
    "data_operations": ["list", "of", "required", "operations"],
    "columns_involved": ["column1", "column2"],
    "filters": {"column": "value or condition"},
    "aggregations": ["sum", "mean", "count", etc.],
    "expected_answer_format": "number|string|boolean|json|base64_image",
    "specific_value_to_find": "description of what value to compute",
    "additional_instructions": "any special notes"
}

Be precise and extract exact column names, filter conditions, and calculation requirements."""

        user_prompt = f"Question: {question_text}"
        if context:
            user_prompt += f"\n\nAdditional context:\n{context}"
        
        try:
            content = await self._call_openai_with_fallback(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                json_mode=True
            )
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            logger.info(f"Interpreted question: {result.get('task_type')}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to interpret question: {e}")
            return {
                "task_type": "unknown",
                "error": str(e),
                "data_operations": [],
                "expected_answer_format": "string"
            }

    async def generate_analysis_code(
        self,
        question_text: str,
        data_preview: Union[str, pd.DataFrame],
        file_type: str = "csv"
    ) -> str:
        """
        Generate Python code to analyze data based on the question.
        
        Tries Gemini first, falls back to OpenAI.
        """
        # Convert DataFrame to string preview if needed
        if isinstance(data_preview, pd.DataFrame):
            df = data_preview
            preview_str = f"""Shape: {df.shape}
Columns: {list(df.columns)}
Dtypes: {df.dtypes.to_dict()}
First 5 rows:
{df.head().to_string()}"""
        else:
            preview_str = data_preview
            df = None
        
        # Try Gemini first
        if self._should_use_gemini() and df is not None:
            code = await self._gemini_helper.generate_analysis_code(question_text, df)
            if code:
                return code
        
        # Fall back to OpenAI
        return await self._generate_analysis_code_openai(question_text, preview_str, file_type)

    async def _generate_analysis_code_openai(
        self,
        question_text: str,
        data_preview: str,
        file_type: str = "csv"
    ) -> str:
        """OpenAI implementation of generate_analysis_code."""
        system_prompt = """You are an expert Python data analyst. Generate clean, efficient Python code to solve data analysis problems.

Rules:
1. Use pandas for data manipulation
2. Use numpy for numerical operations
3. Assume data is already loaded in a variable called 'df'
4. Store the final answer in a variable called 'answer'
5. Do NOT include file loading code - that's handled separately
6. Handle edge cases (missing values, type conversion)
7. Code should be executable as-is
8. Include comments explaining each step

Return ONLY the Python code, no explanations or markdown."""

        user_prompt = f"""Question: {question_text}

Data preview (first rows):
{data_preview}

File type: {file_type}

Generate Python code to compute the answer. The data is already loaded as 'df'.
Store the result in 'answer' variable."""

        try:
            code = await self._call_openai_with_fallback(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            code = code.strip()
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            
            logger.info("Generated analysis code successfully")
            return code.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate code: {e}")
            raise

    async def _format_answer_openai(
        self,
        raw_answer: Any,
        expected_format: str,
        question_context: str
    ) -> Any:
        """OpenAI implementation of format_answer."""
        system_prompt = """You are an assistant that formats answers for automated quiz systems.

Given a raw answer and expected format, return the properly formatted answer.
- For numbers: return exact number without units or formatting
- For strings: return exact string without quotes
- For booleans: return true or false (lowercase)
- For JSON: return valid JSON

Return ONLY the formatted answer, nothing else."""

        user_prompt = f"""Raw answer: {raw_answer}
Expected format: {expected_format}
Question context: {question_context}

Return the properly formatted answer:"""

        try:
            formatted = await self._call_openai_with_fallback(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return self._convert_answer_format(formatted.strip(), expected_format)
            
        except Exception as e:
            logger.error(f"Failed to format answer: {e}")
            return raw_answer

    async def solve_question_directly(
        self,
        question_text: str,
        data_context: Optional[str] = None
    ) -> str:
        """
        Ask LLM to directly solve a question (for simpler questions).
        
        Tries Gemini first, falls back to OpenAI.
        """
        # Try Gemini first
        if self._should_use_gemini():
            result = await self._gemini_helper.chat_completion(
                "You are a precise assistant. Provide only the exact answer - no explanations.",
                f"{question_text}\n\n{data_context}" if data_context else question_text
            )
            if result:
                return result
        
        # Fall back to OpenAI
        return await self._solve_question_openai(question_text, data_context)

    async def _solve_question_openai(
        self,
        question_text: str,
        data_context: Optional[str] = None
    ) -> str:
        """OpenAI implementation of solve_question_directly."""
        system_prompt = """You are a precise assistant that answers data analysis questions.
Provide only the exact answer required - no explanations, no units unless specified, no extra text.
If the answer is a number, provide just the number.
If it's a list, provide comma-separated values.
Be precise and concise."""

        user_prompt = question_text
        if data_context:
            user_prompt = f"{question_text}\n\nData/Context:\n{data_context}"
        
        try:
            answer = await self._call_openai_with_fallback(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            answer = answer.strip()
            logger.info(f"Direct answer: {answer[:100] if answer else 'EMPTY'}...")
            return answer if answer else "test_answer"
            
        except Exception as e:
            logger.error(f"Failed to solve question: {e}")
            raise


# Synchronous wrapper functions
def interpret_question_sync(question_text: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Synchronous wrapper for interpret_question."""
    import asyncio
    helper = LLMHelper()
    return asyncio.run(helper.interpret_question(question_text, context))


def solve_question_sync(question_text: str, data_context: Optional[str] = None) -> str:
    """Synchronous wrapper for solve_question_directly."""
    import asyncio
    helper = LLMHelper()
    return asyncio.run(helper.solve_question_directly(question_text, data_context))


# Singleton instance for convenience
llm_helper = LLMHelper()
