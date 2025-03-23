# main.py - Entry point for the Ultimate Finance Agent

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Import modules for the three main functionalities
from finance_agent.personal_finance import PersonalFinanceAdvisor
from finance_agent.document_analyzer import DocumentAnalyzer
from finance_agent.stock_analyzer import StockMarketAnalyzer
from finance_agent.orchestrator import AgentOrchestrator

# Initialize FastAPI app
app = FastAPI(title="Ultimate Finance Agent")

# Create base models for API requests
class PersonalFinanceRequest(BaseModel):
    situation: str
    income: float
    expenses: Dict[str, float]
    savings: float
    debt: Optional[Dict[str, float]] = None
    goals: List[str]
    risk_tolerance: str
    time_horizon: str

class StockAnalysisRequest(BaseModel):
    tickers: List[str]
    analysis_type: str  # 'technical', 'fundamental', 'sentiment', 'comparative', or 'all'
    time_period: str
    indicators: Optional[List[str]] = None

# Initialize agent modules
personal_finance_advisor = PersonalFinanceAdvisor()
document_analyzer = DocumentAnalyzer()
stock_market_analyzer = StockMarketAnalyzer()

# Initialize the orchestrator
orchestrator = AgentOrchestrator(
    personal_finance_advisor=personal_finance_advisor,
    document_analyzer=document_analyzer,
    stock_market_analyzer=stock_market_analyzer
)

@app.get("/")
async def root():
    return {"message": "Welcome to Ultimate Finance Agent"}

@app.post("/personal-finance/advice")
async def get_personal_finance_advice(request: PersonalFinanceRequest):
    """Get personalized financial advice based on your current situation."""
    response = orchestrator.route_to_personal_finance(
        situation=request.situation,
        income=request.income,
        expenses=request.expenses,
        savings=request.savings,
        debt=request.debt,
        goals=request.goals,
        risk_tolerance=request.risk_tolerance,
        time_horizon=request.time_horizon
    )
    return response

@app.post("/documents/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    summary_type: str = Form("comprehensive")
):
    """Upload and analyze a financial document, with optional specific query."""
    # Save uploaded file temporarily
    temp_file_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    
    with open(temp_file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)
    
    try:
        # Process the document
        if query:
            response = orchestrator.route_to_document_analyzer(
                document_path=temp_file_path,
                query=query
            )
        else:
            response = orchestrator.route_to_document_analyzer(
                document_path=temp_file_path,
                summary_type=summary_type
            )
        return response
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/stocks/analyze")
async def analyze_stocks(request: StockAnalysisRequest):
    """Analyze stock or stocks based on specified parameters."""
    response = orchestrator.route_to_stock_analyzer(
        tickers=request.tickers,
        analysis_type=request.analysis_type,
        time_period=request.time_period,
        indicators=request.indicators
    )
    return response

@app.post("/chat")
async def chat_with_agent(message: str = Form(...), context: Optional[Dict[str, Any]] = Form(None)):
    """General chat endpoint that routes to the appropriate agent based on message content."""
    response = orchestrator.route_message(message, context)
    return response

# If this script is run directly, start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
