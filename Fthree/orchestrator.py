# finance_agent/orchestrator.py

from typing import Dict, List, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END
import re
import json
import os

from personal_finance import PersonalFinanceAdvisor
from document_analyzer import DocumentAnalyzer
from stock_analyzer import StockMarketAnalyzer

class AgentOrchestrator:
    """
    Orchestrates the different finance agents and routes queries to the appropriate one.
    """
    
    def __init__(self, personal_finance_advisor=None, document_analyzer=None, stock_market_analyzer=None):
        # Initialize agent components if not provided
        self.personal_finance_advisor = personal_finance_advisor or PersonalFinanceAdvisor()
        self.document_analyzer = document_analyzer or DocumentAnalyzer()
        self.stock_market_analyzer = stock_market_analyzer or StockMarketAnalyzer()
        
        # Initialize LLM for routing decisions
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Create prompt for classifying user queries
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial agent router. Your job is to analyze a user's message and determine which specialized financial agent should handle it.
            
            Choose EXACTLY ONE of the following categories:
            1. PERSONAL_FINANCE - For personal finance advice, budgeting, savings, retirement planning, etc.
            2. DOCUMENT_ANALYSIS - For analyzing financial documents, reports, or when the user uploads or mentions PDFs
            3. STOCK_ANALYSIS - For stock market analysis, comparing companies, technical/fundamental analysis
            
            Respond with ONLY the category name in capital letters."""),
            ("human", "{query}")
        ])
        
        # Create the routing chain
        self.router_chain = LLMChain(llm=self.llm, prompt=self.router_prompt)
        
        # Set up state graph for more complex workflows
        self.setup_graph()
        
        # Create query extraction prompt
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in financial information extraction. Extract structured data from the user's query.
            
            Based on the query type, extract relevant information and format as JSON:
            
            For PERSONAL_FINANCE queries, extract:
            - income (if mentioned)
            - expenses (as key-value pairs if mentioned)
            - savings (if mentioned)
            - debt (as key-value pairs if mentioned)
            - goals (as a list if mentioned)
            - risk_tolerance (if mentioned: conservative, moderate, aggressive)
            - time_horizon (if mentioned: short, medium, long)
            - situation (general description of their financial situation)
            
            For DOCUMENT_ANALYSIS queries, extract:
            - document_type (if mentioned: 10k, 10q, annual_report, etc.)
            - company (if mentioned)
            - query (specific question about the document)
            - summary_type (if mentioned: brief, comprehensive, key_points, financial_metrics, risk_factors)
            
            For STOCK_ANALYSIS queries, extract:
            - tickers (list of stock symbols mentioned)
            - analysis_type (if mentioned: technical, fundamental, sentiment, comparative)
            - time_period (if mentioned: 1d, 1w, 1m, 3m, 6m, 1y, 2y, 5y)
            - indicators (list of technical indicators if mentioned: SMA, EMA, RSI, MACD, etc.)
            
            Format the response as a clean JSON object with only the relevant fields.
            If information is not provided in the query, omit that field from the JSON.
            """),
            ("human", "Query type: {type}\nQuery: {query}")
        ])
        
        self.extraction_chain = LLMChain(llm=self.llm, prompt=self.extraction_prompt)
    
    def setup_graph(self):
        """Set up the LangGraph for complex multi-step workflows"""
        # Define nodes
        def determine_agent(state):
            """Determine which agent should handle the request"""
            message = state["message"]
            context = state.get("context", {})
            
            # Check if context explicitly specifies an agent
            if context and "agent" in context:
                return {"agent": context["agent"]}
            
            # Otherwise use router chain to classify
            result = self.router_chain.invoke({"query": message})
            agent_type = result["text"].strip()
            return {"agent": agent_type}
        
        def extract_parameters(state):
            """Extract structured parameters from the message"""
            message = state["message"]
            agent_type = state["agent"]
            
            result = self.extraction_chain.invoke({
                "type": agent_type,
                "query": message
            })
            
            try:
                # Try to parse JSON response
                params = self._extract_json(result["text"])
                return {"parameters": params}
            except:
                # If JSON extraction fails, return empty parameters
                return {"parameters": {}}
        
        def route_to_agent(state):
            """Route the query to the appropriate agent"""
            agent_type = state["agent"]
            message = state["message"]
            parameters = state.get("parameters", {})
            context = state.get("context", {})
            
            if agent_type == "PERSONAL_FINANCE":
                result = self.handle_personal_finance_query(message, parameters, context)
            elif agent_type == "DOCUMENT_ANALYSIS":
                result = self.handle_document_analysis_query(message, parameters, context)
            elif agent_type == "STOCK_ANALYSIS":
                result = self.handle_stock_analysis_query(message, parameters, context)
            else:
                # Default to a general response if classification failed
                result = {"response": "I'm not sure how to help with that. Could you provide more details about your financial question?"}
            
            return {"response": result["response"]}
        
        # Create the graph
        workflow = StateGraph({"message": str, "context": Dict, "agent": str, "parameters": Dict, "response": str})
        
        # Add nodes
        workflow.add_node("determine_agent", determine_agent)
        workflow.add_node("extract_parameters", extract_parameters)
        workflow.add_node("route_to_agent", route_to_agent)
        
        # Set up the edges
        workflow.set_entry_point("determine_agent")
        workflow.add_edge("determine_agent", "extract_parameters")
        workflow.add_edge("extract_parameters", "route_to_agent")
        workflow.add_edge("route_to_agent", END)
        
        # Compile the graph
        self.graph = workflow.compile()
    
    def _extract_json(self, text):
        """Helper method to extract JSON from text that might contain markdown or other formatting"""
        # Try to find JSON block in markdown
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no markdown code block, try to find JSON between curly braces
            json_match = re.search(r'({[\s\S]*})', text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON format detected, use the whole text
                json_str = text
        
        # Clean up and parse
        try:
            return json.loads(json_str)
        except:
            # If parsing fails, try to clean up the string further
            clean_str = re.sub(r'[\n\r\t]', ' ', json_str)
            clean_str = re.sub(r'\s+', ' ', clean_str)
            try:
                return json.loads(clean_str)
            except:
                # If all parsing fails, return empty dict
                return {}
    
    def route_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Route a message to the appropriate agent based on its content"""
        if context is None:
            context = {}
            
        # Execute the workflow graph
        result = self.graph.invoke({
            "message": message,
            "context": context,
            "agent": "",
            "parameters": {},
            "response": ""
        })
        
        return {"response": result["response"]}
    
    def handle_personal_finance_query(self, message: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle queries related to personal finance"""
        # Extract parameters from the message if they weren't already extracted
        params = parameters or {}
        
        # Set default values for required parameters
        situation = params.get('situation', message)
        income = params.get('income', 0)
        expenses = params.get('expenses', {})
        savings = params.get('savings', 0)
        debt = params.get('debt')
        goals = params.get('goals')
        risk_tolerance = params.get('risk_tolerance')
        time_horizon = params.get('time_horizon')
        
        # If we don't have basic financial info, use the process_natural_language method
        if income == 0 and not context.get('has_financial_info', False):
            return {"response": self.personal_finance_advisor.process_natural_language(message)}
        
        # Otherwise, use the structured provide_advice method
        result = self.personal_finance_advisor.provide_advice(
            situation=situation,
            income=income,
            expenses=expenses,
            savings=savings,
            debt=debt,
            goals=goals,
            risk_tolerance=risk_tolerance,
            time_horizon=time_horizon
        )
        return result
    
    def handle_document_analysis_query(self, message: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle queries related to document analysis"""
        # Check if we have a document path in the context or parameters
        document_path = context.get("document_path") or parameters.get("document_path")
        
        if document_path and os.path.exists(document_path):
            # Extract query from parameters if available
            query = parameters.get("query", "")
            summary_type = parameters.get("summary_type", "comprehensive")
            
            if query:
                # If we have a document and a specific query about it
                return self.document_analyzer.query_document(document_path=document_path, query=query)
            else:
                # If we have a document but no specific query, generate a summary
                return self.document_analyzer.summarize_document(document_path=document_path, summary_type=summary_type)
        elif context.get("use_agno", False) and document_path:
            # Use AGNO for document analysis if specified
            query = parameters.get("query", message)
            return self.document_analyzer.analyze_with_agno(document_path=document_path, query=query)
        else:
            # No document available, ask the user to upload one
            return {"response": "I'd be happy to analyze a financial document for you. Please upload the document first."}
    
    def handle_stock_analysis_query(self, message: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle queries related to stock market analysis"""
        # Extract tickers from parameters
        tickers = parameters.get("tickers", [])
        
        # If no tickers found in parameters, check context
        if not tickers and "tickers" in context:
            tickers = context["tickers"]
            
        # If still no tickers, try to extract from message using regex
        if not tickers:
            tickers = re.findall(r'\b[A-Z]{1,5}\b', message)
            
        # Filter out common words that might be mistaken for tickers
        common_words = {"I", "A", "THE", "FOR", "IN", "ON", "AT", "TO", "AND", "OR"}
        tickers = [ticker for ticker in tickers if ticker not in common_words]
            
        if tickers:
            # Extract analysis parameters
            analysis_type = parameters.get("analysis_type", "all")
            time_period = parameters.get("time_period", "1y")
            indicators = parameters.get("indicators")
                
            # Run the analysis
            return self.stock_market_analyzer.analyze_stocks(
                tickers=tickers,
                analysis_type=analysis_type,
                time_period=time_period,
                indicators=indicators
            )
        else:
            # No tickers found, ask for clarification
            return {"response": "I'd be happy to analyze stocks for you. Could you specify which company symbols you'd like me to analyze?"}
    
    def route_to_personal_finance(self, situation, income, expenses, savings, debt=None, goals=None, risk_tolerance=None, time_horizon=None):
        """Route a request to the personal finance advisor"""
        result = self.personal_finance_advisor.provide_advice(
            situation=situation,
            income=income,
            expenses=expenses,
            savings=savings,
            debt=debt,
            goals=goals,
            risk_tolerance=risk_tolerance,
            time_horizon=time_horizon
        )
        return result
    
    def route_to_document_analyzer(self, document_path, query=None, summary_type="comprehensive", use_agno=False):
        """Route a request to the document analyzer"""
        if use_agno:
            result = self.document_analyzer.analyze_with_agno(document_path, query)
        elif query:
            result = self.document_analyzer.query_document(document_path, query)
        else:
            result = self.document_analyzer.summarize_document(document_path, summary_type)
        return result
    
    def route_to_stock_analyzer(self, tickers, analysis_type="all", time_period="1y", indicators=None):
        """Route a request to the stock market analyzer"""
        result = self.stock_market_analyzer.analyze_stocks(
            tickers=tickers,
            analysis_type=analysis_type,
            time_period=time_period,
            indicators=indicators
        )
        return result