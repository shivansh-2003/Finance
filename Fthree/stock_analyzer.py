# finance_agent/stock_analyzer.py
from typing import Dict, List, Any, Optional, Union
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import os
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
import matplotlib.pyplot as plt
import io
import base64
# Try to import FinRobot components if available
try:
    from finrobot.functional.analyzer import FinancialAnalyzer
    from finrobot.functional.charting import ChartingTool
    FINROBOT_AVAILABLE = True
except ImportError:
    FINROBOT_AVAILABLE = False

class StockMarketAnalyzer:
    """
    Agent for analyzing stock market data and providing insights.
    """
    
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        
        # Set up API keys
        self.polygon_api_key = os.getenv("POLYGON_API_KEY", "")
        self.rapid_api_key = os.getenv("RAPID_API_KEY", "")
        
        # Set up tools
        self.tools = [
            Tool(
                name="GetStockPriceData",
                func=self._get_stock_price_data,
                description="Gets historical stock price data for a ticker over a specified time period"
            ),
            Tool(
                name="GetStockFundamentals",
                func=self._get_stock_fundamentals,
                description="Gets fundamental financial data for a company including PE ratio, market cap, etc."
            ),
            Tool(
                name="GetCompanyNews",
                func=self._get_company_news,
                description="Gets recent news articles about a company"
            ),
            Tool(
                name="PerformTechnicalAnalysis",
                func=self._perform_technical_analysis,
                description="Performs technical analysis on a stock using indicators like RSI, MACD, etc."
            ),
            Tool(
                name="CalculateComparativeMetrics",
                func=self._calculate_comparative_metrics,
                description="Calculates comparative metrics between multiple stocks"
            ),
            Tool(
                name="GenerateStockCharts",
                func=self._generate_stock_charts,
                description="Generates charts visualizing stock performance or technical indicators"
            ),
        ]
        
        # Add FinRobot tools if available
        if FINROBOT_AVAILABLE:
            self.financial_analyzer = FinancialAnalyzer()
            self.charting_tool = ChartingTool()
            
            self.tools.extend([
                Tool(
                    name="FinRobotFundamentalAnalysis",
                    func=self.financial_analyzer.analyze_fundamentals,
                    description="Uses FinRobot to perform comprehensive fundamental analysis"
                ),
                Tool(
                    name="FinRobotPortfolioAnalysis",
                    func=self.financial_analyzer.analyze_portfolio,
                    description="Uses FinRobot to analyze a portfolio of stocks"
                ),
                Tool(
                    name="FinRobotGenerateAdvancedChart",
                    func=self.charting_tool.create_chart,
                    description="Uses FinRobot to generate advanced financial charts"
                )
            ])
        
        # Create the system prompt for the stock analyzer
        system_prompt = """You are an expert stock market analyst with years of experience analyzing companies and market trends.

Your expertise includes:
- Technical analysis using price charts and indicators
- Fundamental analysis of financial statements and metrics
- Comparative analysis across companies and sectors
- News and sentiment analysis
- Investment strategy formulation

When analyzing stocks:
1. Consider both technical and fundamental factors
2. Look at recent company news and market trends
3. Compare performance against relevant benchmarks
4. Consider macroeconomic factors
5. Assess both risks and potential rewards
6. Provide balanced, evidence-based insights
7. Use data to support your analysis
8. Be clear about the time frame for your analysis
9. Acknowledge areas of uncertainty

Use the available tools to gather data, perform calculations, and generate visualizations before providing your final analysis.
"""
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Create the agent
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    def analyze_stocks(self, tickers: List[str], analysis_type: str = "all", time_period: str = "1y", indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze stocks based on specified parameters.
        
        Parameters:
        - tickers: List of stock ticker symbols
        - analysis_type: Type of analysis to perform (technical, fundamental, sentiment, comparative, or all)
        - time_period: Time period for analysis (e.g., 1d, 1w, 1m, 3m, 6m, 1y, 2y, 5y)
        - indicators: Optional list of technical indicators to include
        
        Returns:
        - Dictionary containing the analysis results
        """
        # Validate and standardize tickers
        tickers = [ticker.upper() for ticker in tickers]
        
        # Validate analysis type
        valid_analysis_types = ["technical", "fundamental", "sentiment", "comparative", "all"]
        if analysis_type not in valid_analysis_types:
            analysis_type = "all"
        
        # Validate indicators
        valid_indicators = ["SMA", "EMA", "RSI", "MACD", "BOLLINGER", "ATR", "OBV", "STOCH"]
        if indicators:
            indicators = [ind.upper() for ind in indicators if ind.upper() in valid_indicators]
        else:
            indicators = ["SMA", "RSI", "MACD"]  # Default indicators
        
        # Create input for the agent
        if len(tickers) == 1:
            ticker_str = tickers[0]
            query = f"Analyze {ticker_str} stock "
        else:
            ticker_str = ", ".join(tickers[:-1]) + " and " + tickers[-1]
            query = f"Compare and analyze {ticker_str} stocks "
        
        if analysis_type != "all":
            query += f"using {analysis_type} analysis "
        
        query += f"over the {time_period} time period. "
        
        if analysis_type in ["technical", "all"]:
            indicator_str = ", ".join(indicators)
            query += f"Include {indicator_str} indicators in the technical analysis. "
        
        query += "Provide a comprehensive analysis with actionable insights. Include relevant charts and visualizations where helpful."
        
        # Execute the agent
        response = self.agent_executor.invoke({"input": query})
        
        # Return the response
        return {"response": response["output"]}
    
    def _get_stock_price_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        """Get historical stock price data using yfinance"""
        try:
            # Download data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            # Basic error checking
            if hist.empty:
                return {"error": f"No data found for {ticker}", "data": None}
            
            # Calculate basic metrics
            metrics = {
                "start_date": hist.index[0].strftime("%Y-%m-%d"),
                "end_date": hist.index[-1].strftime("%Y-%m-%d"),
                "start_price": round(hist["Close"].iloc[0], 2),
                "end_price": round(hist["Close"].iloc[-1], 2),
                "change_pct": round((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100, 2),
                "high": round(hist["High"].max(), 2),
                "low": round(hist["Low"].min(), 2),
                "avg_volume": int(hist["Volume"].mean()),
                "price_volatility": round(hist["Close"].pct_change().std() * 100, 2),  # Daily volatility in percentage
            }
            
            # Convert the data to a dict format that's easier to work with
            price_data = hist.reset_index().to_dict(orient='records')
            
            return {
                "ticker": ticker,
                "period": period,
                "interval": interval,
                "metrics": metrics,
                "price_data": price_data[:10],  # Limit to 10 records for readability
                "total_records": len(price_data)
            }
            
        except Exception as e:
            return {"error": str(e), "data": None}
    
    def _get_stock_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Get fundamental financial data for a company"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get basic info
            info = stock.info
            
            # Extract key financial metrics
            fundamentals = {
                "name": info.get("longName", ticker),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "eps": info.get("trailingEps", 0),
                "beta": info.get("beta", 0),
                "52week_high": info.get("fiftyTwoWeekHigh", 0),
                "52week_low": info.get("fiftyTwoWeekLow", 0),
                "50day_avg": info.get("fiftyDayAverage", 0),
                "200day_avg": info.get("twoHundredDayAverage", 0),
                "profit_margins": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else 0,
                "return_on_equity": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0,
                "return_on_assets": info.get("returnOnAssets", 0) * 100 if info.get("returnOnAssets") else 0,
                "revenue_growth": info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else 0,
            }
            
            # Get financial statements if available
            try:
                # Income Statement
                income_stmt = stock.income_stmt
                if not income_stmt.empty:
                    recent_income = income_stmt.iloc[:, 0]  # Most recent quarter/year
                    fundamentals["total_revenue"] = recent_income.get("Total Revenue", 0)
                    fundamentals["net_income"] = recent_income.get("Net Income", 0)
                
                # Balance Sheet
                balance_sheet = stock.balance_sheet
                if not balance_sheet.empty:
                    recent_balance = balance_sheet.iloc[:, 0]  # Most recent quarter/year
                    fundamentals["total_assets"] = recent_balance.get("Total Assets", 0)
                    fundamentals["total_liabilities"] = recent_balance.get("Total Liabilities Net Minority Interest", 0)
                    fundamentals["total_equity"] = recent_balance.get("Total Equity Gross Minority Interest", 0)
                
                # Cash Flow
                cash_flow = stock.cashflow
                if not cash_flow.empty:
                    recent_cash = cash_flow.iloc[:, 0]  # Most recent quarter/year
                    fundamentals["operating_cash_flow"] = recent_cash.get("Operating Cash Flow", 0)
                    fundamentals["free_cash_flow"] = recent_cash.get("Free Cash Flow", 0)
            except:
                # Financial statements might not be available for all stocks
                pass
                
            return fundamentals
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_company_news(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Get recent news articles about a company"""
        try:
            # Try using Yahoo Finance
            stock = yf.Ticker(ticker)
            news = stock.news
            
            # Format the news articles
            formatted_news = []
            for article in news[:10]:  # Limit to 10 articles
                formatted_news.append({
                    "title": article.get("title", ""),
                    "publisher": article.get("publisher", ""),
                    "link": article.get("link", ""),
                    "publish_time": datetime.fromtimestamp(article.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M"),
                    "type": article.get("type", ""),
                    "summary": article.get("summary", "")[:200] + "..." if len(article.get("summary", "")) > 200 else article.get("summary", "")
                })
            
            # Alternative: use Polygon API if available
            if self.polygon_api_key and not formatted_news:
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                
                url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&published_utc.gte={start_date}&published_utc.lte={end_date}&limit=10&apiKey={self.polygon_api_key}"
                
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    for article in results:
                        formatted_news.append({
                            "title": article.get("title", ""),
                            "publisher": article.get("publisher", {}).get("name", ""),
                            "link": article.get("article_url", ""),
                            "publish_time": article.get("published_utc", ""),
                            "type": "Article",
                            "summary": article.get("description", "")[:200] + "..." if len(article.get("description", "")) > 200 else article.get("description", "")
                        })
            
            # Analyze sentiment of news if we have articles
            sentiment = self._analyze_news_sentiment(formatted_news) if formatted_news else {"sentiment": "neutral", "explanation": "No recent news found"}
            
            return {
                "ticker": ticker,
                "news_count": len(formatted_news),
                "news": formatted_news,
                "sentiment_analysis": sentiment
            }
            
        except Exception as e:
            return {"error": str(e), "news": []}
    
    def _analyze_news_sentiment(self, news_articles: List[Dict[str, str]]) -> Dict[str, str]:
        """Analyze sentiment of news articles using LLM"""
        if not news_articles:
            return {"sentiment": "neutral", "explanation": "No news articles to analyze"}
        
        # Create a simplified list of articles for analysis
        articles_text = ""
        for i, article in enumerate(news_articles[:5], 1):  # Limit to first 5 articles
            articles_text += f"Article {i}: {article['title']}\n"
            articles_text += f"Summary: {article['summary']}\n\n"
        
        # Create a prompt for sentiment analysis
        sentiment_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in financial news sentiment analysis.
            
            Analyze the sentiment of the following news articles about a company.
            Determine if the overall sentiment is positive, negative, or neutral.
            
            Provide:
            1. The overall sentiment (positive, negative, or neutral)
            2. A brief explanation of your assessment
            3. The primary factors influencing your sentiment determination
            """),
            ("human", "{articles}")
        ])
        
        # Create a chain for sentiment analysis
        sentiment_chain = LLMChain(llm=self.llm, prompt=sentiment_prompt)
        
        # Run the chain
        result = sentiment_chain.invoke({"articles": articles_text})
        
        # Process the result
        sentiment_analysis = result["text"]
        
        # Extract the sentiment
        sentiment = "neutral"
        if "positive" in sentiment_analysis.lower():
            sentiment = "positive"
        elif "negative" in sentiment_analysis.lower():
            sentiment = "negative"
        
        return {
            "sentiment": sentiment,
            "explanation": sentiment_analysis
        }
    
    def _generate_comparative_analysis(self, tickers, performance, valuation, rankings):
        """Generate comparative analysis text based on metrics"""
        # Start with a summary
        analysis = f"Comparative Analysis of {', '.join(tickers)}\n\n"
        
        # Performance comparison
        analysis += "Performance Comparison:\n"
        
        # 1-year performance
        if "1y_return" in rankings["performance"] and rankings["performance"]["1y_return"]:
            best_1y = rankings["performance"]["1y_return"][0]
            worst_1y = rankings["performance"]["1y_return"][-1]
            
            analysis += f"- Best 1-year performance: {best_1y} ({performance[best_1y]['1y_return']:.2f}%)\n"
            analysis += f"- Worst 1-year performance: {worst_1y} ({performance[worst_1y]['1y_return']:.2f}%)\n"
        
        # Volatility
        if "volatility" in rankings["performance"] and rankings["performance"]["volatility"]:
            least_volatile = rankings["performance"]["volatility"][0]
            most_volatile = rankings["performance"]["volatility"][-1]
            
            analysis += f"- Least volatile: {least_volatile} (daily volatility: {performance[least_volatile]['volatility']:.2f}%)\n"
            analysis += f"- Most volatile: {most_volatile} (daily volatility: {performance[most_volatile]['volatility']:.2f}%)\n"
        
        # Valuation comparison
        analysis += "\nValuation Comparison:\n"
        
        # P/E Ratio
        if "pe_ratio" in rankings["valuation"] and rankings["valuation"]["pe_ratio"]:
            lowest_pe = rankings["valuation"]["pe_ratio"][0]
            highest_pe = rankings["valuation"]["pe_ratio"][-1]
            
            analysis += f"- Lowest P/E ratio: {lowest_pe} (P/E: {valuation[lowest_pe]['pe_ratio']:.2f})\n"
            analysis += f"- Highest P/E ratio: {highest_pe} (P/E: {valuation[highest_pe]['pe_ratio']:.2f})\n"
        
        # Dividend yield
        if "dividend_yield" in rankings["valuation"] and rankings["valuation"]["dividend_yield"]:
            highest_div = rankings["valuation"]["dividend_yield"][0]
            lowest_div = rankings["valuation"]["dividend_yield"][-1]
            
            analysis += f"- Highest dividend yield: {highest_div} (yield: {valuation[highest_div]['dividend_yield']:.2f}%)\n"
            analysis += f"- Lowest dividend yield: {lowest_div} (yield: {valuation[lowest_div]['dividend_yield']:.2f}%)\n"
        
        # Market cap comparison
        market_caps = {ticker: data["market_cap"] for ticker, data in valuation.items() if "market_cap" in data and data["market_cap"]}
        if market_caps:
            largest = max(market_caps.items(), key=lambda x: x[1])[0]
            smallest = min(market_caps.items(), key=lambda x: x[1])[0]
            
            # Format market cap in billions
            largest_cap = valuation[largest]["market_cap"] / 1e9
            smallest_cap = valuation[smallest]["market_cap"] / 1e9
            
            analysis += f"- Largest company by market cap: {largest} (${largest_cap:.2f}B)\n"
            analysis += f"- Smallest company by market cap: {smallest} (${smallest_cap:.2f}B)\n"
        
        # Overall comparison summary
        analysis += "\nOverall Comparison Summary:\n"
        
        # Try to identify the best investment based on simple criteria
        best_value = None
        best_growth = None
        best_dividend = None
        best_overall = None
        
        # Best value (lowest P/E with positive growth)
        if "pe_ratio" in rankings["valuation"] and rankings["valuation"]["pe_ratio"]:
            for ticker in rankings["valuation"]["pe_ratio"]:
                if ticker in performance and performance[ticker].get("1y_return", 0) > 0:
                    best_value = ticker
                    break
        
        # Best growth (highest 1y return with reasonable P/E)
        if "1y_return" in rankings["performance"] and rankings["performance"]["1y_return"]:
            for ticker in rankings["performance"]["1y_return"]:
                if ticker in valuation and valuation[ticker].get("pe_ratio", 100) < 50:  # P/E under 50 as a simple filter
                    best_growth = ticker
                    break
        
        # Best dividend (highest yield with positive performance)
        if "dividend_yield" in rankings["valuation"] and rankings["valuation"]["dividend_yield"]:
            for ticker in rankings["valuation"]["dividend_yield"]:
                if ticker in performance and performance[ticker].get("1y_return", 0) > 0:
                    best_dividend = ticker
                    break
        
        # Add findings to analysis
        if best_value:
            analysis += f"- Best value investment: {best_value} (P/E: {valuation[best_value]['pe_ratio']:.2f}, 1-year return: {performance[best_value]['1y_return']:.2f}%)\n"
        
        if best_growth:
            analysis += f"- Best growth investment: {best_growth} (1-year return: {performance[best_growth]['1y_return']:.2f}%, P/E: {valuation[best_growth].get('pe_ratio', 'N/A')})\n"
        
        if best_dividend:
            analysis += f"- Best dividend investment: {best_dividend} (yield: {valuation[best_dividend]['dividend_yield']:.2f}%, 1-year return: {performance[best_dividend]['1y_return']:.2f}%)\n"
        
        return analysis
    
    def _generate_stock_charts(self, ticker: str, chart_type: str = "price", period: str = "1y", indicators: List[str] = None) -> Dict[str, Any]:
        """Generate stock charts using matplotlib via base64 encoding"""
        try:
            # Check if FinRobot's charting is available and use it if possible
            if FINROBOT_AVAILABLE:
                return self._generate_finrobot_chart(ticker, chart_type, period, indicators)
            
            # Otherwise, generate basic charts
            
            # Get stock data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return {"error": f"No data found for {ticker}", "chart": None}
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Generate appropriate chart based on type
            if chart_type == "price":
                # Plot price chart
                plt.plot(hist.index, hist["Close"], label=f"{ticker} Close Price")
                
                # Add moving averages if requested
                if indicators and "SMA" in indicators:
                    plt.plot(hist.index, hist["Close"].rolling(window=50).mean(), label="50-day SMA", linestyle="--")
                    plt.plot(hist.index, hist["Close"].rolling(window=200).mean(), label="200-day SMA", linestyle="-.")
                
                plt.title(f"{ticker} Stock Price ({period})")
                plt.ylabel("Price ($)")
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            elif chart_type == "volume":
                # Plot volume chart
                plt.bar(hist.index, hist["Volume"], alpha=0.7, color="blue")
                
                # Add moving average for volume
                plt.plot(hist.index, hist["Volume"].rolling(window=20).mean(), color="red", label="20-day MA")
                
                plt.title(f"{ticker} Trading Volume ({period})")
                plt.ylabel("Volume")
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            elif chart_type == "rsi":
                # Calculate RSI
                delta = hist["Close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                # Plot RSI
                plt.plot(hist.index, rsi, label="RSI(14)", color="purple")
                
                # Add overbought/oversold lines
                plt.axhline(y=70, color="red", linestyle="--", alpha=0.5)
                plt.axhline(y=30, color="green", linestyle="--", alpha=0.5)
                
                plt.title(f"{ticker} Relative Strength Index ({period})")
                plt.ylabel("RSI")
                plt.ylim(0, 100)
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            elif chart_type == "macd":
                # Calculate MACD
                ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
                ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                histogram = macd_line - signal_line
                
                # Plot MACD
                plt.plot(hist.index, macd_line, label="MACD Line", color="blue")
                plt.plot(hist.index, signal_line, label="Signal Line", color="red")
                plt.bar(hist.index, histogram, label="Histogram", alpha=0.5, color="green")
                
                plt.title(f"{ticker} MACD ({period})")
                plt.ylabel("MACD")
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            graph = base64.b64encode(image_png).decode('utf-8')
            
            return {
                "ticker": ticker,
                "chart_type": chart_type,
                "period": period,
                "chart": f"data:image/png;base64,{graph}"
            }
            
        except Exception as e:
            return {"error": str(e), "chart": None}
    
    def _generate_finrobot_chart(self, ticker: str, chart_type: str, period: str, indicators: List[str]):
        """Generate charts using FinRobot's charting tool if available"""
        try:
            # Map chart types to FinRobot chart types
            chart_mapping = {
                "price": "line",
                "candlestick": "candlestick",
                "volume": "volume",
                "rsi": "rsi",
                "macd": "macd"
            }
            
            # Map period to FinRobot period
            period_mapping = {
                "1d": "1d",
                "5d": "5d",
                "1m": "1mo",
                "3m": "3mo",
                "6m": "6mo",
                "1y": "1y",
                "2y": "2y",
                "5y": "5y",
                "max": "max"
            }
            
            # Create indicator config
            indicator_config = {}
            if indicators:
                for ind in indicators:
                    if ind == "SMA":
                        indicator_config["sma"] = [20, 50, 200]
                    elif ind == "EMA":
                        indicator_config["ema"] = [12, 26]
                    elif ind == "BOLLINGER":
                        indicator_config["bollinger"] = True
                    elif ind == "RSI":
                        indicator_config["rsi"] = True
                    elif ind == "MACD":
                        indicator_config["macd"] = True
            
            # Generate chart
            chart_result = self.charting_tool.create_chart(
                ticker=ticker,
                chart_type=chart_mapping.get(chart_type, "line"),
                period=period_mapping.get(period, "1y"),
                indicators=indicator_config,
                title=f"{ticker} {chart_type.capitalize()} Chart ({period})"
            )
            
            return chart_result
            
        except Exception as e:
            return {"error": str(e), "chart": None}