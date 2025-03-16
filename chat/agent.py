import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.x import XTools

load_dotenv()

YAHOO_FINANCE_KEY = os.getenv("YAHOO_FINANCE_KEY")
TAVILY_KEY = os.getenv("TAVILY_KEY")
CLAUDE_KEY = os.getenv("CLAUDE_KEY")

# 🔹 Web Agent - Fetches financial news & insights
web_agent = Agent(
    name="Web Agent",
    role="Search the web for financial news and trends",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools(),TavilyTools(api_key=TAVILY_KEY, search=True, search_depth="advanced", format="markdown")],
    instructions="Always include sources and recent updates",
    show_tool_calls=True,
    markdown=True,
)

# 🔹 Finance Agent - Retrieves stock market & investment data
finance_agent = Agent(
    name="Finance Agent",
    role="Fetch financial data and market analysis",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions="Use tables to display financial data. Provide investment suggestions based on trends.",
    show_tool_calls=True,
    markdown=True,
)

# 🔹 Country-Specific Finance Assistant
finance_assistant = Agent(
    team=[web_agent, finance_agent],
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "Answer financial questions based on the user's country (India or USA).",
        "Provide personalized investment advice based on current market trends.",
        "Consider economic conditions like inflation, recession, and policy changes.",
        "Always include sources and explain investment risks."
    ],
    show_tool_calls=True,
    markdown=True,
)

# 🔹 User Interaction (CLI-Based)
print("\n💰 Welcome to Your Personal Finance Assistant! (Supports India & USA) 💰")

while True:
    user_input = input("\n🔹 Ask a financial question (or type 'exit' to quit): ")
    
    if user_input.lower() == "exit":
        print("📉 Exiting the finance assistant. Have a great day! 🚀")
        break

    print("\n🔍 Fetching financial insights...\n")
    
    # 🔹 Process the question with the AI Finance Assistant
    finance_assistant.print_response(user_input, stream=True)