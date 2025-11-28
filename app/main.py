"""FastAPI Application - LLM Analysis Quiz API."""

import asyncio
import logging
from typing import Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from app.core.config import get_settings, validate_secret
from app.services.quiz_solver import QuizSolver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuizRequest(BaseModel):
    """Request model for the /quiz endpoint."""
    email: str = Field(..., description="User email address")
    secret: str = Field(..., description="Authentication secret")
    url: str = Field(..., description="Quiz URL to solve")
    
    class Config:
        extra = "allow"
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class QuizResultResponse(BaseModel):
    """Response model for completed quiz results."""
    status: str
    total_time: float
    questions_attempted: int
    results: list


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting LLM Analysis Quiz API")
    yield
    logger.info("Shutting down LLM Analysis Quiz API")


app = FastAPI(
    title="LLM Analysis Quiz API",
    description="API for automated quiz solving with LLM-assisted data analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


active_tasks: dict = {}


@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {"name": "LLM Analysis Quiz API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/quiz", response_model=QuizResultResponse)
async def solve_quiz(request: QuizRequest):
    """Main quiz solving endpoint."""
    logger.info(f"Received quiz request for URL: {request.url}")
    
    settings = get_settings()
    if not validate_secret(request.secret):
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    try:
        solver = QuizSolver(email=request.email, secret=request.secret)
        result = await solver.solve(request.url)
        
        return QuizResultResponse(
            status=result.get("status", "completed"),
            total_time=result.get("total_time", 0),
            questions_attempted=result.get("questions_attempted", 0),
            results=result.get("results", [])
        )
    except Exception as e:
        logger.error(f"Quiz solving failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
