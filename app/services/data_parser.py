"""
Data Parser Utility Module

This module provides utilities for parsing various data file formats
commonly encountered in data analysis quizzes:
- CSV files (with various encodings and delimiters)
- Excel files (xlsx, xls)
- JSON files (records, arrays, nested)
- PDF files (text extraction)
- Text files

Design philosophy: Be flexible and try multiple parsing strategies
rather than failing on the first attempt.
"""

import io
import json
import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def parse_csv(
    file_content: Union[str, bytes],
    filename: str = "data.csv"
) -> Optional[pd.DataFrame]:
    """
    Parse CSV content with automatic encoding and delimiter detection.
    
    Tries multiple combinations of encodings and delimiters to find
    what works for the given file.
    
    Args:
        file_content: Raw file content (string or bytes).
        filename: Original filename for logging.
        
    Returns:
        DataFrame if successfully parsed, None otherwise.
    """
    # Convert bytes to string if needed
    if isinstance(file_content, bytes):
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        content_str = None
        
        for enc in encodings:
            try:
                content_str = file_content.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if content_str is None:
            logger.error(f"Could not decode {filename} with any known encoding")
            return None
    else:
        content_str = file_content
    
    # Try different delimiters
    delimiters = [',', ';', '\t', '|', ' ']
    
    for delimiter in delimiters:
        try:
            df = pd.read_csv(
                io.StringIO(content_str),
                sep=delimiter,
                on_bad_lines='skip'
            )
            
            # Sanity check: should have at least 2 columns or be non-empty
            if len(df.columns) > 1 or len(df) > 0:
                logger.info(f"Parsed {filename} with delimiter '{delimiter}'")
                return df
                
        except Exception as e:
            logger.debug(f"Failed to parse with delimiter '{delimiter}': {e}")
            continue
    
    # Last resort: try pandas auto-detection
    try:
        df = pd.read_csv(
            io.StringIO(content_str),
            sep=None,
            engine='python',
            on_bad_lines='skip'
        )
        return df
    except Exception as e:
        logger.error(f"All CSV parsing attempts failed for {filename}: {e}")
        return None


def parse_excel(
    file_content: bytes,
    filename: str = "data.xlsx"
) -> Optional[pd.DataFrame]:
    """
    Parse Excel file content.
    
    Args:
        file_content: Raw Excel file bytes.
        filename: Original filename for logging.
        
    Returns:
        DataFrame if successfully parsed, None otherwise.
    """
    try:
        # Use BytesIO for Excel files
        df = pd.read_excel(io.BytesIO(file_content))
        logger.info(f"Parsed Excel file {filename}: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to parse Excel file {filename}: {e}")
        return None


def parse_json(
    file_content: Union[str, bytes],
    filename: str = "data.json"
) -> Optional[pd.DataFrame]:
    """
    Parse JSON content into a DataFrame.
    
    Handles various JSON structures:
    - Array of records: [{"a": 1}, {"a": 2}]
    - Single object: {"a": 1, "b": 2}
    - Nested structures
    
    Args:
        file_content: Raw JSON content.
        filename: Original filename for logging.
        
    Returns:
        DataFrame if successfully parsed, None otherwise.
    """
    # Convert bytes to string if needed
    if isinstance(file_content, bytes):
        file_content = file_content.decode('utf-8')
    
    try:
        data = json.loads(file_content)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Array of records
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Check if it's a records-style dict
            if all(isinstance(v, list) for v in data.values()):
                df = pd.DataFrame(data)
            else:
                # Single record
                df = pd.DataFrame([data])
        else:
            logger.error(f"Unknown JSON structure in {filename}")
            return None
        
        logger.info(f"Parsed JSON file {filename}: {df.shape}")
        return df
        
    except json.JSONDecodeError as e:
        # Try pandas JSON reader which is more flexible
        try:
            df = pd.read_json(io.StringIO(file_content))
            return df
        except Exception:
            logger.error(f"Failed to parse JSON file {filename}: {e}")
            return None


def parse_pdf(
    file_content: bytes,
    filename: str = "document.pdf"
) -> str:
    """
    Extract text from PDF file.
    
    Uses pdfplumber for better text extraction.
    Falls back to PyPDF2 if pdfplumber fails.
    
    Args:
        file_content: Raw PDF bytes.
        filename: Original filename for logging.
        
    Returns:
        Extracted text content.
    """
    text = ""
    
    # Try pdfplumber first (better extraction)
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if text.strip():
            logger.info(f"Extracted {len(text)} chars from PDF {filename}")
            return text.strip()
            
    except Exception as e:
        logger.debug(f"pdfplumber failed for {filename}: {e}")
    
    # Fallback to PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_content))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        logger.info(f"Extracted {len(text)} chars from PDF {filename} (PyPDF2)")
        return text.strip()
        
    except Exception as e:
        logger.error(f"Failed to extract PDF text from {filename}: {e}")
        return ""


def parse_file(
    file_content: Union[str, bytes],
    filename: str
) -> Union[pd.DataFrame, str, None]:
    """
    Universal file parser that detects type and parses accordingly.
    
    Args:
        file_content: Raw file content.
        filename: Filename (used for extension detection).
        
    Returns:
        DataFrame for data files, string for text/PDF, None on failure.
    """
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if ext == 'csv':
        return parse_csv(file_content, filename)
    elif ext in ['xlsx', 'xls']:
        return parse_excel(file_content if isinstance(file_content, bytes) else file_content.encode(), filename)
    elif ext == 'json':
        return parse_json(file_content, filename)
    elif ext == 'pdf':
        return parse_pdf(file_content if isinstance(file_content, bytes) else file_content.encode(), filename)
    elif ext == 'txt':
        if isinstance(file_content, bytes):
            return file_content.decode('utf-8', errors='ignore')
        return file_content
    elif ext == 'parquet':
        try:
            return pd.read_parquet(io.BytesIO(file_content if isinstance(file_content, bytes) else file_content.encode()))
        except Exception as e:
            logger.error(f"Failed to parse parquet {filename}: {e}")
            return None
    else:
        # Try CSV as default
        logger.info(f"Unknown extension '{ext}', trying CSV parser")
        return parse_csv(file_content, filename)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame by handling common issues.
    
    Operations:
    - Strip whitespace from string columns
    - Convert numeric strings to numbers
    - Handle common NA representations
    - Remove completely empty rows/columns
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
        # Replace common NA representations
        df[col] = df[col].replace(['', 'nan', 'NaN', 'NA', 'N/A', 'null', 'None'], np.nan)
        
        # Try to convert to numeric
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    
    return df


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of a DataFrame for LLM context.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Dictionary with summary statistics.
    """
    summary = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "sample_values": {},
        "statistics": {}
    }
    
    # Sample values for each column
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            summary["sample_values"][col] = non_null.head(3).tolist()
    
    # Statistics for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        summary["statistics"] = numeric_df.describe().to_dict()
    
    return summary
