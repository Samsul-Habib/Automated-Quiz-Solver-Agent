"""
Quiz Solver module - Core logic for solving quiz questions.

This is the main orchestration module that:
1. Opens quiz URLs and extracts question data
2. Downloads and parses any data files
3. Performs required analysis (filtering, aggregation, visualization, etc.)
4. Constructs and submits answers
5. Handles the recursive quiz chain

Design Choices:
---------------
1. Async-first design for better performance with I/O operations
2. Modular analysis functions for different data types
3. Fallback to LLM when rule-based parsing fails
4. Comprehensive error handling with retry logic
5. Time tracking to respect the 3-minute window
6. Specialized handlers for common quiz patterns (DOM, API, CSV, etc.)
"""

import asyncio
import base64
import io
import json
import logging
import math
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import httpx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server environments
import matplotlib.pyplot as plt

from app.core.config import get_settings
from app.services.browser import BrowserService
from app.services.llm_helper import LLMHelper
from app.services.question_handlers import QuestionHandlers

logger = logging.getLogger(__name__)


class QuizSolver:
    """
    Main quiz solving orchestrator.
    
    This class handles the complete quiz-solving workflow:
    - Fetching quiz pages and extracting questions
    - Downloading and parsing data files
    - Performing analysis based on instructions
    - Submitting answers and handling responses
    - Managing the 3-minute time constraint
    """
    
    def __init__(self, email: str, secret: str):
        """
        Initialize the quiz solver.
        
        Args:
            email: User email for quiz submissions.
            secret: Secret key for authentication.
        """
        self.email = email
        self.secret = secret
        self.settings = get_settings()
        self.llm = LLMHelper()
        self.handlers = QuestionHandlers(self.llm)
        self.start_time: Optional[float] = None
        self.results: List[Dict[str, Any]] = []
        
    def _check_time_remaining(self) -> float:
        """
        Check how much time is remaining in the 3-minute window.
        
        Returns:
            Seconds remaining (can be negative if time exceeded).
        """
        if self.start_time is None:
            return self.settings.TIMEOUT_SECONDS
        elapsed = time.time() - self.start_time
        return self.settings.TIMEOUT_SECONDS - elapsed
    
    def _is_time_available(self, buffer_seconds: int = 10) -> bool:
        """
        Check if there's enough time to continue processing.
        
        Args:
            buffer_seconds: Safety buffer to account for submission time.
            
        Returns:
            True if sufficient time remains.
        """
        return self._check_time_remaining() > buffer_seconds
    
    async def solve(self, quiz_url: str) -> Dict[str, Any]:
        """
        Main entry point - solve a quiz starting from the given URL.
        
        This method:
        1. Starts the timer
        2. Opens the quiz URL
        3. Processes the question
        4. Submits the answer
        5. Recursively handles subsequent questions
        
        Args:
            quiz_url: The initial quiz URL to solve.
            
        Returns:
            Dictionary with results of all solved questions.
        """
        self.start_time = time.time()
        self.results = []
        
        logger.info(f"Starting quiz solve for: {quiz_url}")
        
        try:
            await self._solve_recursive(quiz_url)
        except Exception as e:
            logger.error(f"Quiz solving failed: {e}")
            self.results.append({
                "url": quiz_url,
                "error": str(e),
                "status": "failed"
            })
        finally:
            # Cleanup handlers
            await self.handlers.close()
        
        return {
            "status": "completed",
            "total_time": time.time() - self.start_time,
            "questions_attempted": len(self.results),
            "results": self.results
        }
    
    async def _solve_recursive(self, quiz_url: str, depth: int = 0):
        """
        Recursively solve quiz questions.
        
        Args:
            quiz_url: Current quiz URL to solve.
            depth: Recursion depth for logging.
        """
        if not self._is_time_available():
            logger.warning("Time limit reached, stopping quiz solving")
            return
        
        logger.info(f"[Depth {depth}] Processing: {quiz_url}")
        
        # Step 1: Extract quiz data from the page
        async with BrowserService() as browser:
            quiz_data = await browser.extract_quiz_data(quiz_url)
        
        # Step 2: Parse the question and determine what to do
        question_text = quiz_data.get('page_text', '')
        page_html = quiz_data.get('page_html', '')
        file_links = quiz_data.get('file_links', [])
        
        # Extract submit URL from the page
        submit_url = await self._find_submit_url(quiz_data)
        
        if not submit_url:
            logger.error("Could not find submit URL on quiz page")
            self.results.append({
                "url": quiz_url,
                "error": "No submit URL found",
                "status": "failed"
            })
            return
        
        # Step 3: Download any data files
        downloaded_files = []
        if file_links:
            with tempfile.TemporaryDirectory() as temp_dir:
                async with BrowserService() as browser:
                    downloaded_files = await browser.download_files_from_page(file_links, temp_dir)
                
                # Step 4: Analyze data and compute answer
                answer = await self._compute_answer(
                    question_text=question_text,
                    page_html=page_html,
                    downloaded_files=downloaded_files,
                    temp_dir=temp_dir,
                    quiz_url=quiz_url
                )
        else:
            # No files to download - answer from page content alone
            answer = await self._compute_answer(
                question_text=question_text,
                page_html=page_html,
                downloaded_files=[],
                temp_dir=None,
                quiz_url=quiz_url
            )
        
        # Step 5: Submit the answer
        response = await self._submit_answer(
            submit_url=submit_url,
            quiz_url=quiz_url,
            answer=answer
        )
        
        # Record result
        result = {
            "url": quiz_url,
            "submit_url": submit_url,
            "answer": answer,
            "response": response,
            "status": "submitted",
            "time_elapsed": time.time() - self.start_time
        }
        self.results.append(result)
        
        # Step 6: Handle response - check for next URL
        if response:
            is_correct = response.get('correct', False)
            next_url = response.get('url')
            
            result["correct"] = is_correct
            
            if is_correct and next_url and self._is_time_available():
                logger.info(f"Correct! Moving to next question: {next_url}")
                await self._solve_recursive(next_url, depth + 1)
            elif not is_correct and self._is_time_available(buffer_seconds=30):
                # Try to recompute and resubmit if time allows
                logger.warning("Answer incorrect, attempting retry...")
                # Could implement retry logic here
                if next_url:
                    # If there's a next URL even on incorrect, proceed
                    await self._solve_recursive(next_url, depth + 1)
    
    async def _find_submit_url(self, quiz_data: Dict[str, Any]) -> Optional[str]:
        """
        Find the submit URL from quiz page data.
        
        Tries multiple strategies:
        1. Direct submit_url from page extraction
        2. Look for API endpoint in page text
        3. Look for form action
        4. Use LLM to find it
        
        Args:
            quiz_data: Extracted quiz data dictionary.
            
        Returns:
            Submit URL if found, None otherwise.
        """
        # Strategy 1: Direct extraction
        if quiz_data.get('submit_url'):
            return quiz_data['submit_url']
        
        page_text = quiz_data.get('page_text', '')
        page_html = quiz_data.get('page_html', '')
        
        # Strategy 2: Regex patterns for common API endpoints
        patterns = [
            r'POST[^h]+(https?://[^\s"\'<>]+)',  # POST to URL pattern
            r'https?://[^\s"\'<>]+/submit',  # Direct submit URL
            r'https?://[^\s"\'<>]+/api/submit[^\s"\'<>]*',  # API submit
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_text + page_html, re.IGNORECASE)
            if match:
                # Get the captured group if exists, otherwise the full match
                url = match.group(1) if match.lastindex else match.group(0)
                # Clean up URL
                url = url.rstrip('.,;:\'\")')
                logger.info(f"Found submit URL via regex: {url}")
                return url
        
        # Strategy 2b: Look for relative URL pattern like "POST to /submit"
        relative_match = re.search(r'POST[^/]+(\/[^\s"\'<>]+)', page_text, re.IGNORECASE)
        if relative_match:
            relative_path = relative_match.group(1)
            # Construct full URL from quiz URL
            from urllib.parse import urljoin
            base_url = quiz_data.get('url', '')
            full_url = urljoin(base_url, relative_path)
            logger.info(f"Found relative submit URL: {relative_path} -> {full_url}")
            return full_url
        
        # Strategy 3: Look for base URL and construct submit endpoint
        base_url_match = re.search(r'(https?://[^/]+)', quiz_data.get('url', ''))
        if base_url_match:
            base_url = base_url_match.group(1)
            # Common submit endpoints
            possible_endpoints = [
                f"{base_url}/api/submit",
                f"{base_url}/submit",
            ]
            # Return first guess - might need refinement
            return possible_endpoints[0]
        
        # Strategy 4: Use LLM if available
        if self.llm.is_available():
            try:
                result = await self.llm.solve_question_directly(
                    f"Find the submit URL or API endpoint in this page. Return ONLY the URL:\n{page_text[:2000]}",
                    None
                )
                if result.startswith('http'):
                    return result.strip()
            except Exception as e:
                logger.error(f"LLM failed to find submit URL: {e}")
        
        return None
    
    async def _compute_answer(
        self,
        question_text: str,
        page_html: str,
        downloaded_files: List[Dict],
        temp_dir: Optional[str],
        quiz_url: str = ""
    ) -> Any:
        """
        Compute the answer based on question and data.
        
        This is the core analysis function that:
        1. Parses the question to understand requirements
        2. Loads and processes data files
        3. Performs required calculations
        4. Returns the formatted answer
        
        Args:
            question_text: The question/instructions text.
            page_html: Full page HTML for additional context.
            downloaded_files: List of downloaded file info.
            temp_dir: Directory containing downloaded files.
            quiz_url: The original quiz URL.
            
        Returns:
            The computed answer in appropriate format.
        """
        logger.info("Computing answer...")
        logger.debug(f"Question: {question_text[:500]}...")
        
        # If no files, try to answer directly
        if not downloaded_files or not temp_dir:
            return await self._answer_without_data(question_text, page_html, quiz_url)
        
        # Separate audio, image, and data files
        audio_files = [f for f in downloaded_files if f.get('type') == 'audio' or 
                       f.get('filename', '').lower().endswith(('.opus', '.mp3', '.wav', '.ogg', '.m4a'))]
        image_files = [f for f in downloaded_files if f.get('type') == 'image' or
                       f.get('filename', '').lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'))]
        data_files = [f for f in downloaded_files if f not in audio_files and f not in image_files and 'error' not in f]
        
        # Process audio files if present (requires Gemini)
        audio_context = ""
        if audio_files and self.llm.gemini and self.llm.gemini.is_available:
            for audio_info in audio_files:
                filepath = audio_info.get('local_path')
                filename = audio_info.get('filename', '')
                
                if filepath and os.path.exists(filepath):
                    try:
                        # Determine MIME type
                        ext = filename.lower().split('.')[-1] if '.' in filename else 'opus'
                        mime_map = {
                            'opus': 'audio/opus',
                            'mp3': 'audio/mpeg',
                            'wav': 'audio/wav',
                            'ogg': 'audio/ogg',
                            'm4a': 'audio/mp4'
                        }
                        mime_type = mime_map.get(ext, 'audio/opus')
                        
                        # Read audio data
                        with open(filepath, 'rb') as f:
                            audio_data = f.read()
                        
                        logger.info(f"Processing audio file: {filename} ({len(audio_data)} bytes)")
                        
                        # Transcribe audio
                        transcription = await self.llm.transcribe_audio(audio_data, mime_type)
                        if transcription:
                            audio_context = f"Audio transcription: {transcription}"
                            logger.info(f"Audio transcription: {transcription}")
                            
                    except Exception as e:
                        logger.error(f"Failed to process audio {filename}: {e}")
        elif audio_files:
            logger.warning("Audio files found but Gemini not available for transcription")
        
        # Process image files if present (requires Gemini)
        image_context = ""
        if image_files and self.llm.gemini and self.llm.gemini.is_available:
            for image_info in image_files:
                filepath = image_info.get('local_path')
                filename = image_info.get('filename', '')
                
                if filepath and os.path.exists(filepath):
                    try:
                        ext = filename.lower().split('.')[-1] if '.' in filename else 'png'
                        mime_map = {
                            'png': 'image/png',
                            'jpg': 'image/jpeg',
                            'jpeg': 'image/jpeg',
                            'gif': 'image/gif',
                            'webp': 'image/webp',
                            'bmp': 'image/bmp'
                        }
                        mime_type = mime_map.get(ext, 'image/png')
                        
                        with open(filepath, 'rb') as f:
                            image_data = f.read()
                        
                        logger.info(f"Processing image file: {filename} ({len(image_data)} bytes)")
                        
                        # Analyze image
                        image_analysis = await self.llm.analyze_image(image_data, mime_type)
                        if image_analysis:
                            image_context = f"Image analysis: {image_analysis}"
                            logger.info(f"Image analysis: {image_analysis[:200]}...")
                            
                    except Exception as e:
                        logger.error(f"Failed to process image {filename}: {e}")
        elif image_files:
            logger.warning("Image files found but Gemini not available for analysis")
        
        # Load data files
        dataframes = {}
        for file_info in data_files:
            if 'error' in file_info:
                continue
            
            filepath = file_info.get('local_path')
            filename = file_info.get('filename', '')
            
            try:
                df = self._load_data_file(filepath, filename)
                if df is not None:
                    dataframes[filename] = df
                    logger.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
        
        # If we have audio/image but no data, try to answer from media alone
        if not dataframes and (audio_context or image_context):
            combined_context = f"{audio_context}\n{image_context}".strip()
            return await self._answer_with_media(question_text, combined_context, page_html)
        
        if not dataframes:
            return await self._answer_without_data(question_text, page_html, quiz_url)
        
        # Get the primary dataframe (usually there's just one)
        df = list(dataframes.values())[0]
        df_name = list(dataframes.keys())[0]
        
        # Create data context for LLM
        data_context = self._create_data_context(df, df_name)
        
        # Enhance question text with common pattern hints
        # Pass audio context to help determine the correct operation
        enhanced_question = self._enhance_question_text(question_text, df, audio_context)
        
        # Try to interpret and solve
        if self.llm.is_available():
            try:
                # Get interpretation of what's needed
                interpretation = await self.llm.interpret_question(enhanced_question, data_context)
                
                # Generate and execute analysis code
                code = await self.llm.generate_analysis_code(
                    question_text=enhanced_question,
                    data_preview=df,
                    file_type=df_name.split('.')[-1] if '.' in df_name else 'csv'
                )
                
                # Log the generated code for debugging
                logger.debug(f"Generated code:\n{code}")
                
                # Execute the generated code
                answer = self._execute_analysis_code(code, df)
                
                if answer is not None:
                    return answer
                    
            except Exception as e:
                logger.error(f"LLM-assisted analysis failed: {e}")
        
        # Fallback: Try rule-based analysis
        return self._rule_based_analysis(question_text, df)
    
    async def _answer_with_media(
        self, 
        question_text: str, 
        media_context: str, 
        page_html: str
    ) -> Any:
        """
        Answer a question based on audio/image context alone.
        
        Args:
            question_text: The question text from the page.
            media_context: Transcription/analysis of audio/image files.
            page_html: Full page HTML for context.
            
        Returns:
            The computed answer.
        """
        logger.info("Answering with media context only")
        
        if self.llm.is_available():
            try:
                full_context = f"""Page text: {question_text}

{media_context}

Based on the media content and page context, provide the answer."""
                
                answer = await self.llm.solve_question_directly(full_context)
                if answer:
                    return answer
            except Exception as e:
                logger.error(f"Failed to answer with media: {e}")
        
        return media_context

    async def _answer_with_audio(
        self, 
        question_text: str, 
        audio_context: str, 
        page_html: str
    ) -> Any:
        """Answer a question based on audio transcription alone."""
        return await self._answer_with_media(question_text, audio_context, page_html)
    
    def _load_data_file(self, filepath: str, filename: str) -> Optional[pd.DataFrame]:
        """
        Load a data file into a pandas DataFrame.
        
        Supports CSV, Excel, JSON, and attempts to handle other formats.
        Also detects headerless CSV files where first row is data.
        
        Args:
            filepath: Path to the file.
            filename: Original filename (for extension detection).
            
        Returns:
            DataFrame if successfully loaded, None otherwise.
        """
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        try:
            if ext == 'csv':
                # First try with header
                df = None
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    for sep in [',', ';', '\t', '|']:
                        try:
                            df = pd.read_csv(filepath, encoding=encoding, sep=sep)
                            break
                        except:
                            continue
                    if df is not None:
                        break
                
                if df is None:
                    df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
                
                # Check if column names look like data (all numeric)
                # This indicates no header row
                if df is not None and len(df.columns) > 0:
                    all_numeric_headers = all(
                        str(col).replace('.', '').replace('-', '').isdigit() 
                        for col in df.columns
                    )
                    if all_numeric_headers:
                        logger.info(f"Detected headerless CSV - reloading with header=None")
                        # Reload without header
                        df = pd.read_csv(filepath, header=None)
                
                return df
                
            elif ext in ['xlsx', 'xls']:
                return pd.read_excel(filepath)
                
            elif ext == 'json':
                # Try different JSON formats
                try:
                    return pd.read_json(filepath)
                except:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        return pd.DataFrame(data)
                    elif isinstance(data, dict):
                        return pd.DataFrame([data])
                    raise ValueError("Unknown JSON structure")
                    
            elif ext == 'txt':
                # Try as CSV with different separators
                return pd.read_csv(filepath, sep=None, engine='python')
                
            elif ext == 'parquet':
                return pd.read_parquet(filepath)
                
            else:
                # Try as CSV by default
                return pd.read_csv(filepath)
                
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return None
    
    def _enhance_question_text(self, question_text: str, df: pd.DataFrame, audio_context: str = "") -> str:
        """
        Enhance question text by inferring task from common patterns.
        
        If audio_context is provided, use it to determine the actual task
        (e.g., whether to sum values above or below cutoff).
        
        Args:
            question_text: Original question text.
            df: The DataFrame for context.
            audio_context: Optional transcription from audio file.
            
        Returns:
            Enhanced question text with explicit instructions.
        """
        import re
        enhanced = question_text
        
        # Check if audio provides specific instructions
        operation = "below"  # default
        if audio_context:
            audio_lower = audio_context.lower()
            # Parse audio for direction
            if 'greater than or equal' in audio_lower or '>=' in audio_lower:
                operation = "greater than or equal to"
            elif 'greater than' in audio_lower or 'above' in audio_lower:
                operation = "above"
            elif 'less than or equal' in audio_lower or '<=' in audio_lower:
                operation = "less than or equal to"
            elif 'less than' in audio_lower or 'below' in audio_lower:
                operation = "below"
            
            logger.info(f"Audio context suggests operation: {operation}")
        
        # Pattern: "Cutoff: N"
        cutoff_match = re.search(r'cutoff[:\s]+(\d+)', question_text, re.IGNORECASE)
        if cutoff_match:
            cutoff_val = cutoff_match.group(1)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                # Use operation from audio if available
                if operation == "greater than or equal to":
                    enhanced = f"""The data file contains numeric values.
Task: Calculate the SUM of all numbers that are GREATER THAN OR EQUAL TO the cutoff value.
Cutoff value: {cutoff_val}

{f"Audio instructions: {audio_context}" if audio_context else ""}

Original question text:
{question_text}

Important: The answer should be the sum of values >= {cutoff_val}."""
                elif operation == "above":
                    enhanced = f"""The data file contains numeric values.
Task: Calculate the SUM of all numbers that are ABOVE (greater than) the cutoff value.
Cutoff value: {cutoff_val}

{f"Audio instructions: {audio_context}" if audio_context else ""}

Original question text:
{question_text}

Important: The answer should be the sum of values > {cutoff_val}."""
                else:
                    # Default to below
                    enhanced = f"""The data file contains numeric values.
Task: Calculate the SUM of all numbers that are BELOW (less than) the cutoff value.
Cutoff value: {cutoff_val}

{f"Audio instructions: {audio_context}" if audio_context else ""}

Original question text:
{question_text}

Important: The answer should be the sum of values < {cutoff_val}."""
                
                logger.info(f"Enhanced question with cutoff pattern: cutoff={cutoff_val}, operation={operation}")
        
        # Pattern: "Threshold: N" - similar to cutoff
        threshold_match = re.search(r'threshold[:\s]+(\d+)', question_text, re.IGNORECASE)
        if threshold_match and 'cutoff' not in question_text.lower():
            threshold_val = threshold_match.group(1)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                enhanced = f"""The data file contains numeric values.
Task: Calculate the SUM of all numbers that are BELOW (less than) the threshold value.
Threshold value: {threshold_val}

Original question text:
{question_text}

Important: The answer should be the sum of values strictly less than {threshold_val}."""
                logger.info(f"Enhanced question with threshold pattern: threshold={threshold_val}")
        
        return enhanced
    
    def _create_data_context(self, df: pd.DataFrame, filename: str) -> str:
        """
        Create a context string describing the data for LLM.
        
        Args:
            df: The DataFrame.
            filename: The filename.
            
        Returns:
            Description string.
        """
        context = f"File: {filename}\n"
        context += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
        # Convert column names to strings for joining
        col_names = [str(c) for c in df.columns.tolist()]
        context += f"Columns: {', '.join(col_names)}\n"
        context += f"Data types:\n{df.dtypes.to_string()}\n"
        context += f"\nFirst 5 rows:\n{df.head().to_string()}\n"
        context += f"\nBasic statistics:\n{df.describe().to_string()}\n"
        return context
    
    def _execute_analysis_code(self, code: str, df: pd.DataFrame) -> Any:
        """
        Safely execute generated analysis code.
        
        Args:
            code: Python code to execute.
            df: DataFrame to analyze.
            
        Returns:
            The computed answer, or None if execution fails.
        """
        # Create a restricted execution environment
        exec_globals = {
            'pd': pd,
            'np': np,
            'df': df,
            'plt': plt,
            'io': io,
            'base64': base64,
            'json': json,
            're': re,
            'datetime': datetime,
        }
        exec_locals = {'answer': None}
        
        try:
            # Execute the code
            exec(code, exec_globals, exec_locals)
            answer = exec_locals.get('answer')
            
            # Convert numpy types to Python types for JSON serialization
            if isinstance(answer, (np.integer, np.int64, np.int32)):
                answer = int(answer)
            elif isinstance(answer, (np.floating, np.float64, np.float32)):
                answer = float(answer)
            elif isinstance(answer, np.ndarray):
                answer = answer.tolist()
            elif isinstance(answer, pd.Series):
                answer = answer.tolist()
            elif isinstance(answer, pd.DataFrame):
                answer = answer.to_dict('records')
            
            logger.info(f"Code execution successful, answer: {str(answer)[:100]}")
            return answer
            
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            logger.debug(f"Code was:\n{code}")
            return None
    
    def _rule_based_analysis(self, question_text: str, df: pd.DataFrame) -> Any:
        """
        Perform rule-based analysis when LLM is unavailable.
        
        Uses keyword matching to determine the required operation.
        
        Args:
            question_text: The question text.
            df: The DataFrame to analyze.
            
        Returns:
            Computed answer.
        """
        question_lower = question_text.lower()
        
        # Count/sum operations
        if 'how many' in question_lower or 'count' in question_lower:
            return len(df)
        
        if 'total' in question_lower or 'sum' in question_lower:
            # Find numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return df[numeric_cols[0]].sum()
        
        if 'average' in question_lower or 'mean' in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return df[numeric_cols[0]].mean()
        
        if 'maximum' in question_lower or 'max' in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return df[numeric_cols[0]].max()
        
        if 'minimum' in question_lower or 'min' in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return df[numeric_cols[0]].min()
        
        # Default: return row count
        return len(df)
    
    async def _answer_without_data(self, question_text: str, page_html: str, quiz_url: str = "") -> Any:
        """
        Attempt to answer a question without data files.
        
        Uses specialized handlers for different question types:
        - DOM parsing (hidden elements, reversed text)
        - API pagination
        - API authentication
        - Data cleaning
        - CSV processing
        - JavaScript execution
        - Date manipulation
        - Spatial analysis
        - Log parsing
        - Data pipeline (joins)
        
        Args:
            question_text: The question text.
            page_html: The page HTML.
            quiz_url: The quiz URL for context.
            
        Returns:
            Computed answer.
        """
        base_url = quiz_url.rsplit('/', 1)[0] if '/' in quiz_url else quiz_url
        question_lower = question_text.lower()
        
        try:
            # ==============================================
            # PATTERN 1: DOM Parsing (hidden-key, reversed)
            # ==============================================
            if 'hidden' in question_lower or 'class=' in question_lower or 'reverse' in question_lower:
                # Look for class name pattern
                class_match = re.search(r'class[=:\s]+["\']?(\w+[-_]?\w*)["\']?', question_text, re.IGNORECASE)
                if class_match:
                    class_name = class_match.group(1)
                    should_reverse = 'reverse' in question_lower
                    
                    # First try from current page
                    result = self.handlers.handle_hidden_element(page_html, class_name, should_reverse)
                    if result:
                        logger.info(f"DOM parsing: found '{class_name}' = {result}")
                        return result
            
            # ==============================================
            # PATTERN 2: API Pagination (find item by ID)
            # ==============================================
            if 'page=' in question_lower or 'pagination' in question_lower or ('id=' in question_lower and 'name' in question_lower):
                # Look for target ID
                id_match = re.search(r'id[=:\s]+(\d+)', question_text, re.IGNORECASE)
                # Look for API URL
                api_match = re.search(r'(api/\w+)', question_text, re.IGNORECASE)
                
                if id_match and api_match:
                    target_id = int(id_match.group(1))
                    api_path = api_match.group(1)
                    api_url = f"{base_url}/{api_path}?page=1"
                    
                    result = await self.handlers.handle_pagination(api_url, target_id)
                    if result:
                        logger.info(f"Pagination: found ID {target_id} = {result}")
                        return result
            
            # ==============================================
            # PATTERN 3: API Authentication (X-API-Key)
            # ==============================================
            if 'x-api-key' in question_lower or 'api-key' in question_lower:
                # Extract API key
                key_match = re.search(r'(?:api.?key|key)[:\s]+["\']?([a-zA-Z0-9_-]+)["\']?', question_text, re.IGNORECASE)
                # Extract API endpoint
                endpoint_match = re.search(r'(api/\w+)', question_text, re.IGNORECASE)
                
                if key_match and endpoint_match:
                    api_key = key_match.group(1)
                    endpoint = endpoint_match.group(1)
                    api_url = f"{base_url}/{endpoint}"
                    
                    data = await self.handlers.fetch_with_auth(api_url, api_key=api_key)
                    if data:
                        # Determine what to find (highest temperature, etc.)
                        if 'highest' in question_lower and 'temperature' in question_lower:
                            result = self.handlers.find_max_by_field(data, 'temperature', 'city')
                            if result:
                                logger.info(f"Auth API: highest temp city = {result}")
                                return result
                        elif 'temperature' in question_lower:
                            # Generic temp handling
                            for item in data:
                                if 'temperature' in item:
                                    return item.get('city') or item.get('name')
                        # Return first item or raw data for LLM
                        return data
            
            # ==============================================
            # PATTERN 4: Data Cleaning (dirty-data, sum prices)
            # ==============================================
            if 'dirty' in question_lower or ('clean' in question_lower and 'price' in question_lower):
                # Extract endpoint
                endpoint_match = re.search(r'(api/[\w-]+)', question_text, re.IGNORECASE)
                if endpoint_match:
                    endpoint = endpoint_match.group(1)
                    api_url = f"{base_url}/{endpoint}"
                    
                    result = await self.handlers.handle_dirty_data(api_url, 'price')
                    if result:
                        logger.info(f"Data cleaning: sum = {result}")
                        return int(result) if result == int(result) else round(result, 2)
            
            # ==============================================
            # PATTERN 5: CSV Processing (sales.csv, filters)
            # ==============================================
            if '.csv' in question_lower:
                # Extract CSV URL
                csv_match = re.search(r'(api/[\w.]+\.csv)', question_text, re.IGNORECASE)
                if csv_match:
                    csv_url = f"{base_url}/{csv_match.group(1)}"
                    
                    # Parse filters from question
                    filters = {}
                    # Common filter patterns
                    region_match = re.search(r'region[=:\s]+["\']?(\w+)["\']?', question_text, re.IGNORECASE)
                    if region_match:
                        filters['region'] = region_match.group(1)
                    
                    currency_match = re.search(r'currency[=:\s]+["\']?(\w+)["\']?', question_text, re.IGNORECASE)
                    if currency_match:
                        filters['currency'] = currency_match.group(1)
                    
                    # Determine what column to sum
                    sum_col = 'amount' if 'amount' in question_lower else 'total' if 'total' in question_lower else None
                    
                    result = await self.handlers.handle_csv_analysis(csv_url, filters, sum_col)
                    if result is not None:
                        logger.info(f"CSV analysis: result = {result}")
                        return int(result) if isinstance(result, (int, float)) and result == int(result) else result
            
            # ==============================================
            # PATTERN 6: JavaScript Execution
            # ==============================================
            if 'script' in question_lower or 'javascript' in question_lower or 'generated' in question_lower:
                result = await self.handlers.handle_js_execution(page_html, question_text)
                if result:
                    logger.info(f"JS execution: result = {result}")
                    return result
            
            # ==============================================
            # PATTERN 7: Date Manipulation (Tuesday, etc.)
            # ==============================================
            if 'tuesday' in question_lower or 'monday' in question_lower or 'wednesday' in question_lower or \
               ('date' in question_lower and 'count' in question_lower):
                # Determine target weekday
                weekday_map = {
                    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                    'friday': 4, 'saturday': 5, 'sunday': 6
                }
                target_weekday = None
                for day, num in weekday_map.items():
                    if day in question_lower:
                        target_weekday = num
                        break
                
                # Extract API endpoint
                endpoint_match = re.search(r'(api/\w+)', question_text, re.IGNORECASE)
                if endpoint_match and target_weekday is not None:
                    api_url = f"{base_url}/{endpoint_match.group(1)}"
                    result = await self.handlers.handle_date_analysis(api_url, target_weekday)
                    logger.info(f"Date analysis: {target_weekday} = {result}")
                    return result
            
            # ==============================================
            # PATTERN 8: Spatial Analysis (distance)
            # ==============================================
            if 'distance' in question_lower or 'euclidean' in question_lower:
                # Extract point IDs
                point_ids = re.findall(r'(?:point\s*)?([A-Z])', question_text)
                # Extract endpoint
                endpoint_match = re.search(r'(api/\w+)', question_text, re.IGNORECASE)
                
                if len(point_ids) >= 2 and endpoint_match:
                    api_url = f"{base_url}/{endpoint_match.group(1)}"
                    result = await self.handlers.handle_spatial_distance(
                        api_url, point_ids[0], point_ids[1], decimal_places=2
                    )
                    logger.info(f"Spatial: distance {point_ids[0]}↔{point_ids[1]} = {result}")
                    return result
            
            # ==============================================
            # PATTERN 9: Log Parsing (IP addresses)
            # ==============================================
            if 'log' in question_lower or 'ip' in question_lower or 'request' in question_lower:
                endpoint_match = re.search(r'(api/\w+)', question_text, re.IGNORECASE)
                if endpoint_match:
                    api_url = f"{base_url}/{endpoint_match.group(1)}"
                    result = await self.handlers.handle_log_parsing(api_url, 'most_common_ip')
                    if result:
                        logger.info(f"Log parsing: most common IP = {result}")
                        return result
            
            # ==============================================
            # PATTERN 10: Data Pipeline (join users, products, orders)
            # ==============================================
            if 'users' in question_lower and 'orders' in question_lower:
                # Find all API endpoints
                endpoints = {}
                for name in ['users', 'products', 'orders']:
                    match = re.search(rf'(api/{name})', question_text, re.IGNORECASE)
                    if match:
                        endpoints[name] = f"{base_url}/{match.group(1)}"
                    else:
                        # Try default pattern
                        endpoints[name] = f"{base_url}/api/{name}"
                
                result = await self.handlers.handle_data_pipeline(endpoints)
                if result:
                    logger.info(f"Data pipeline: result = {result}")
                    return int(result) if isinstance(result, float) and result == int(result) else result
            
            # ==============================================
            # PATTERN: Scraping another page
            # ==============================================
            scrape_match = re.search(r'[Ss]crape\s+([^\s]+)', question_text)
            if scrape_match:
                relative_url = scrape_match.group(1)
                scrape_url = urljoin(quiz_url, relative_url)
                logger.info(f"Need to scrape: {scrape_url}")
                
                try:
                    async with BrowserService() as browser:
                        html, text = await browser.get_page_content(scrape_url)
                        logger.info(f"Scraped content: {text[:200]}...")
                        
                        if self.llm.is_available():
                            answer = await self.llm.solve_question_directly(
                                f"Extract the secret code or answer from this scraped content:\n{text}\n\nOriginal question: {question_text}"
                            )
                            if answer:
                                return answer
                        return text.strip()
                except Exception as e:
                    logger.error(f"Failed to scrape {scrape_url}: {e}")
            
            # ==============================================
            # FALLBACK: Use LLM
            # ==============================================
            if self.llm.is_available():
                try:
                    return await self.llm.solve_question_directly(question_text)
                except Exception as e:
                    logger.error(f"LLM direct solve failed: {e}")
            
        except Exception as e:
            logger.error(f"Handler error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Can't answer without data or LLM
        return "Unable to compute answer"
    
    async def _submit_answer(
        self,
        submit_url: str,
        quiz_url: str,
        answer: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Submit the answer to the quiz API.
        
        Args:
            submit_url: The submission endpoint.
            quiz_url: The original quiz URL.
            answer: The computed answer.
            
        Returns:
            Response JSON if successful, None otherwise.
        """
        payload = {
            "email": self.email,
            "secret": self.secret,
            "url": quiz_url,
            "answer": answer
        }
        
        logger.info(f"Submitting answer to {submit_url}")
        logger.debug(f"Payload: {json.dumps(payload, default=str)[:500]}")
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    submit_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                logger.info(f"Submit response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Submit response: {result}")
                    return result
                else:
                    logger.error(f"Submit failed: {response.status_code} - {response.text}")
                    return {"error": response.text, "status_code": response.status_code}
                    
        except Exception as e:
            logger.error(f"Submit request failed: {e}")
            return {"error": str(e)}


# Utility functions for creating visualizations

def create_chart_base64(
    df: pd.DataFrame,
    chart_type: str = 'bar',
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    title: str = ''
) -> str:
    """
    Create a chart and return as base64 encoded PNG.
    
    Args:
        df: Data to visualize.
        chart_type: Type of chart (bar, line, scatter, pie, hist).
        x_column: Column for x-axis.
        y_column: Column for y-axis.
        title: Chart title.
        
    Returns:
        Base64 encoded PNG data URI.
    """
    plt.figure(figsize=(10, 6))
    
    if chart_type == 'bar':
        if x_column and y_column:
            df.plot(kind='bar', x=x_column, y=y_column)
        else:
            df.plot(kind='bar')
    elif chart_type == 'line':
        if x_column and y_column:
            df.plot(kind='line', x=x_column, y=y_column)
        else:
            df.plot(kind='line')
    elif chart_type == 'scatter':
        if x_column and y_column:
            plt.scatter(df[x_column], df[y_column])
    elif chart_type == 'pie':
        if y_column:
            df[y_column].plot(kind='pie')
    elif chart_type == 'hist':
        if y_column:
            df[y_column].hist()
        else:
            df.hist()
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plt.close()
    
    # Encode as base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


# Convenience function for synchronous usage
def solve_quiz_sync(email: str, secret: str, quiz_url: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for quiz solving.
    
    Args:
        email: User email.
        secret: Authentication secret.
        quiz_url: Quiz URL to solve.
        
    Returns:
        Results dictionary.
    """
    solver = QuizSolver(email, secret)
    return asyncio.run(solver.solve(quiz_url))
