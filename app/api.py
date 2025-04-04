"""
Vercel deployment entry point for fastapi_chatbot.py
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import base64
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the app from fastapi_chatbot.py as api_app
from fastapi_chatbot import app as api_app, UploadCSVModel, AskModel, SaveDashboardModel

# Create the main app that will be used by Vercel
app = FastAPI(title="Data Visualization API - Vercel")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the original API app
# This will route all requests to the original app
app.mount("/api", api_app)

# Root endpoint for health check
@app.get("/")
async def root():
    """Health check endpoint for Vercel"""
    return {
        "status": "online",
        "message": "Data Visualization API is running on Vercel",
        "documentation": "/docs",
        "api_route": "/api"
    }

# For Vercel serverless deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000) 