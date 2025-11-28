"""
Configuration module for the LLM Analysis Quiz application.

This module handles all configuration and secrets management.
All sensitive values are read from environment variables to ensure
security best practices - no secrets are hardcoded.

Design choice: Using Pydantic BaseSettings for automatic env var loading
and validation. This provides type safety and clear documentation of
required configuration.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Attributes:
        QUIZ_EMAIL: The email address to use when submitting quiz answers.
        QUIZ_SECRET: The secret key for authentication with the quiz API.
        OPENAI_API_KEY: Optional OpenAI API key for LLM-assisted question solving.
        OPENAI_MODEL: The OpenAI model to use (default: gpt-4o-mini for cost efficiency).
        APP_SECRET: Secret key to verify incoming requests to our API.
        DEBUG: Enable debug mode for verbose logging.
        TIMEOUT_SECONDS: Maximum time (in seconds) for the entire quiz solving process.
    """
    
    # Required configuration
    QUIZ_EMAIL: str = ""  # TODO: Set via environment variable
    QUIZ_SECRET: str = ""  # TODO: Set via environment variable
    APP_SECRET: str = ""   # Secret to verify incoming POST requests
    
    # Optional LLM configuration - Gemini (primary)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.0-flash"  # Fast, multimodal (text, audio, images)
    
    # Optional LLM configuration - OpenAI (fallback)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None  # Custom base URL (e.g., https://aipipe.org/openai/v1)
    OPENAI_MODEL: str = "gpt-4o-mini"  # Cost-effective model for analysis
    OPENAI_FALLBACK_MODEL: Optional[str] = None  # Fallback model if primary fails
    
    # LLM Provider selection: "gemini", "openai", or "auto" (try gemini first, then openai)
    LLM_PROVIDER: str = "auto"
    
    # Application settings
    DEBUG: bool = False
    TIMEOUT_SECONDS: int = 180  # 3 minutes as per spec
    
    # Browser settings
    BROWSER_HEADLESS: bool = True  # Run browser without GUI
    BROWSER_TIMEOUT: int = 30000   # 30 seconds timeout for browser operations
    
    class Config:
        """Pydantic config to load from .env file if present."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Using lru_cache ensures we only load settings once,
    improving performance and consistency.
    
    Returns:
        Settings: The application settings instance.
    """
    return Settings()


def validate_secret(provided_secret: str) -> bool:
    """
    Validate the provided secret against the configured APP_SECRET.
    
    Args:
        provided_secret: The secret provided in the incoming request.
        
    Returns:
        bool: True if the secret matches, False otherwise.
    """
    settings = get_settings()
    return provided_secret == settings.APP_SECRET


# Convenience function to get settings without import overhead
settings = get_settings()
