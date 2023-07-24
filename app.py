from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textsummarizer.pipeline.prediction import PredictionPipeline

text:str = "What is Text Summarization"

app = FastAPI()

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")
