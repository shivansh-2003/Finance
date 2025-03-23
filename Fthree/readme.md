---
noteId: "3462a510081911f0ae384d80017bbbd4"
tags: []

---

# Ultimate Finance Agent

A comprehensive AI-powered platform for financial analysis that incorporates three core modules:
1. Personal Finance Advisor
2. Financial Document Analyzer
3. Stock Market Analyzer

## Architecture

The Ultimate Finance Agent uses a layered architecture with the following components:

![Ultimate Finance Agent Architecture](images/architecture.png)

- **User Interaction Layer**: Provides interfaces for users to interact with the agent
- **Orchestration Layer**: Routes requests to the appropriate specialized agent
- **Specialized Agent Modules**: Three core modules for different financial analysis domains
- **Data Sources Layer**: Connects to various financial data sources and APIs

## Features

### Personal Finance Advisor
- Provides tailored financial advice based on individual situations
- Optimizes budgets and expense allocation
- Creates customized investment plans
- Suggests debt repayment strategies
- Develops savings plans to meet financial goals
- Provides tax optimization recommendations

### Financial Document Analyzer
- Analyzes financial documents using RAG (Retrieval Augmented Generation)
- Processes various document types (10-K, 10-Q, annual reports, research reports)
- Extracts key financial metrics and insights
- Identifies and analyzes risk factors
- Answers specific questions about documents
- Generates comprehensive or focused summaries

### Stock Market Analyzer
- Performs comprehensive stock analysis (technical, fundamental, sentiment)
- Generates comparative analysis between multiple stocks
- Creates visualization charts for price trends and indicators
- Analyzes financial news sentiment for stocks
- Provides actionable investment insights
- Supports various technical indicators (SMA, RSI, MACD, etc.)

## Installation

### Prerequisites
- Python 3.9+
- Required API keys:
  - OpenAI API key
  - (Optional) Financial API keys (Polygon, RapidAPI, etc.)

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/finance-ai/ultimate-finance-agent.git
cd ultimate-finance-agent
```

2. Install the package:
```bash
pip install -e .
```

3. Create a configuration file:
```bash
cp config_example.json config.json
```

4. Edit `config.json` to add your API keys.

## Usage

### Command Line Interface

The agent can be used from the command line:

```bash
# Get personal finance advice
finance-agent personal-finance --income 5000 --savings 10000 --expenses '{"rent": 1500, "food": 500, "utilities": 200}' --situation "Looking to save for a house down payment"

# Analyze a financial document
finance-agent analyze-document --document-path annual_report.pdf --summary-type comprehensive

# Analyze stocks
finance-agent analyze-stocks --tickers AAPL,MSFT,GOOGL --analysis-type comparative --time-period 1y

# Start interactive chat
finance-agent chat
```

### Python API

```python
from finance_agent import create_agent

# Create agent
agent = create_agent(openai_api_key="your-api-key")

# Personal finance advice
result = agent.route_to_personal_finance(
    situation="Looking to save for a house down payment",
    income=5000,
    expenses={"rent": 1500, "food": 500, "utilities": 200},
    savings=10000,
    goals=["Buy a house in 5 years", "Retire by 60"]
)
print(result["response"])

# Document analysis
result = agent.route_to_document_analyzer(
    document_path="annual_report.pdf",
    summary_type="comprehensive"
)
print(result["response"])

# Stock analysis
result = agent.route_to_stock_analyzer(
    tickers=["AAPL", "MSFT", "GOOGL"],
    analysis_type="comparative",
    time_period="1y"
)
print(result["response"])

# Chat interface
response = agent.route_message("What stocks should I consider for a long-term investment?")
print(response["response"])
```

## Configuration

The agent can be configured using a JSON configuration file:

```json
{
    "OPENAI_API_KEY": "your-openai-api-key",
    "POLYGON_API_KEY": "your-polygon-api-key",
    "RAPID_API_KEY": "your-rapid-api-key",
    "FRED_API_KEY": "your-fred-api-key"
}
```

## Advanced Usage

### Integrating with FinRobot

The Ultimate Finance Agent can optionally integrate with [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) for enhanced capabilities:

```python
from finance_agent import create_agent

# Create agent with FinRobot integration
agent = create_agent(use_finrobot=True)
```

### Using Agno for Document Analysis

For advanced document analysis, you can use the Agno integration:

```bash
finance-agent analyze-document --document-path annual_report.pdf --use-agno
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

The financial advice and analysis provided by this agent is for informational purposes only and should not be considered as professional financial advice. Always consult with a qualified financial advisor before making investment decisions.
