# finance_agent/utils.py

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import requests
import yfinance as yf
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('finance_agent')

def load_api_keys(config_file: str = "config.json") -> Dict[str, str]:
    """
    Load API keys from configuration file.
    
    Parameters:
    - config_file: Path to the config file containing API keys
    
    Returns:
    - Dictionary of API keys
    """
    try:
        if not os.path.exists(config_file):
            logger.warning(f"Config file {config_file} not found. Using environment variables instead.")
            return {}
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Set environment variables for keys
        for key, value in config.items():
            if key.endswith('_API_KEY'):
                os.environ[key] = value
        
        return config
    except Exception as e:
        logger.error(f"Error loading API keys: {e}")
        return {}

def format_currency(amount: float) -> str:
    """
    Format a number as a currency string.
    
    Parameters:
    - amount: The amount to format
    
    Returns:
    - Formatted currency string
    """
    if amount >= 1e12:
        return f"${amount / 1e12:.2f}T"
    elif amount >= 1e9:
        return f"${amount / 1e9:.2f}B"
    elif amount >= 1e6:
        return f"${amount / 1e6:.2f}M"
    elif amount >= 1e3:
        return f"${amount / 1e3:.2f}K"
    else:
        return f"${amount:.2f}"

def format_percentage(value: float, include_sign: bool = True) -> str:
    """
    Format a value as a percentage string.
    
    Parameters:
    - value: The value to format as percentage
    - include_sign: Whether to include + sign for positive values
    
    Returns:
    - Formatted percentage string
    """
    if include_sign and value > 0:
        return f"+{value:.2f}%"
    else:
        return f"{value:.2f}%"

def get_current_date() -> str:
    """
    Get current date in YYYY-MM-DD format.
    
    Returns:
    - Current date string
    """
    return datetime.now().strftime("%Y-%m-%d")

def validate_ticker(ticker: str) -> bool:
    """
    Validate if a ticker symbol exists.
    
    Parameters:
    - ticker: Stock ticker symbol to validate
    
    Returns:
    - True if valid, False otherwise
    """
    try:
        
        stock = yf.Ticker(ticker)
        info = stock.info
        return 'symbol' in info and info['symbol'] == ticker
    except:
        return False

def extract_financial_metrics(text: str) -> Dict[str, Any]:
    """
    Extract financial metrics from text using regex patterns.
    
    Parameters:
    - text: Text containing financial metrics
    
    Returns:
    - Dictionary of extracted metrics
    """
    import re
    
    metrics = {}
    
    # Revenue pattern (e.g., "revenue of $1.2 billion" or "revenue: $1.2B")
    revenue_pattern = r'revenue(?:\s+of)?\s+\$?(\d+(?:\.\d+)?)\s*(million|billion|trillion|M|B|T)?'
    revenue_match = re.search(revenue_pattern, text, re.IGNORECASE)
    if revenue_match:
        amount = float(revenue_match.group(1))
        unit = revenue_match.group(2)
        if unit:
            if unit.lower() in ['million', 'm']:
                amount *= 1e6
            elif unit.lower() in ['billion', 'b']:
                amount *= 1e9
            elif unit.lower() in ['trillion', 't']:
                amount *= 1e12
        metrics['revenue'] = amount
    
    # EPS pattern (e.g., "EPS of $1.23" or "earnings per share: $1.23")
    eps_pattern = r'(?:eps|earnings per share)(?:\s+of)?\s+\$?(\d+\.\d+)'
    eps_match = re.search(eps_pattern, text, re.IGNORECASE)
    if eps_match:
        metrics['eps'] = float(eps_match.group(1))
    
    # P/E pattern (e.g., "P/E ratio of 15.2" or "PE: 15.2")
    pe_pattern = r'(?:p/?e|price[- ]to[- ]earnings)(?:\s+(?:ratio|of))?\s+(\d+(?:\.\d+)?)'
    pe_match = re.search(pe_pattern, text, re.IGNORECASE)
    if pe_match:
        metrics['pe_ratio'] = float(pe_match.group(1))
    
    # Market cap pattern (e.g., "market cap of $1.2 billion" or "market capitalization: $1.2B")
    mcap_pattern = r'market\s+cap(?:italization)?(?:\s+of)?\s+\$?(\d+(?:\.\d+)?)\s*(million|billion|trillion|M|B|T)?'
    mcap_match = re.search(mcap_pattern, text, re.IGNORECASE)
    if mcap_match:
        amount = float(mcap_match.group(1))
        unit = mcap_match.group(2)
        if unit:
            if unit.lower() in ['million', 'm']:
                amount *= 1e6
            elif unit.lower() in ['billion', 'b']:
                amount *= 1e9
            elif unit.lower() in ['trillion', 't']:
                amount *= 1e12
        metrics['market_cap'] = amount
    
    return metrics

def fetch_economic_indicators() -> Dict[str, Any]:
    """
    Fetch current economic indicators using public APIs.
    
    Returns:
    - Dictionary of economic indicators
    """
    indicators = {}
    
    try:
        # Try to fetch FRED data using API if available
        fred_api_key = os.environ.get('FRED_API_KEY')
        if fred_api_key:
            # Get GDP growth rate
            gdp_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=GDP&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=2"
            gdp_response = requests.get(gdp_url)
            if gdp_response.status_code == 200:
                gdp_data = gdp_response.json()
                if 'observations' in gdp_data and len(gdp_data['observations']) >= 2:
                    current_gdp = float(gdp_data['observations'][0]['value'])
                    previous_gdp = float(gdp_data['observations'][1]['value'])
                    gdp_growth = ((current_gdp / previous_gdp) - 1) * 100
                    indicators['gdp_growth'] = gdp_growth
            
            # Get unemployment rate
            unemployment_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=1"
            unemployment_response = requests.get(unemployment_url)
            if unemployment_response.status_code == 200:
                unemployment_data = unemployment_response.json()
                if 'observations' in unemployment_data and len(unemployment_data['observations']) >= 1:
                    indicators['unemployment_rate'] = float(unemployment_data['observations'][0]['value'])
            
            # Get inflation rate (CPI)
            inflation_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=13"
            inflation_response = requests.get(inflation_url)
            if inflation_response.status_code == 200:
                inflation_data = inflation_response.json()
                if 'observations' in inflation_data and len(inflation_data['observations']) >= 13:
                    current_cpi = float(inflation_data['observations'][0]['value'])
                    year_ago_cpi = float(inflation_data['observations'][12]['value'])
                    inflation_rate = ((current_cpi / year_ago_cpi) - 1) * 100
                    indicators['inflation_rate'] = inflation_rate
            
            # Get Fed Funds Rate
            fed_rate_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=1"
            fed_rate_response = requests.get(fed_rate_url)
            if fed_rate_response.status_code == 200:
                fed_rate_data = fed_rate_response.json()
                if 'observations' in fed_rate_data and len(fed_rate_data['observations']) >= 1:
                    indicators['fed_funds_rate'] = float(fed_rate_data['observations'][0]['value'])
    except Exception as e:
        logger.error(f"Error fetching economic indicators: {e}")
    
    # Fall back to hardcoded recent values if API fails
    if not indicators:
        indicators = {
            'gdp_growth': 2.1,
            'unemployment_rate': 3.7,
            'inflation_rate': 3.1,
            'fed_funds_rate': 5.25
        }
    
    return indicators

def generate_report_filename(report_type: str, subject: str) -> str:
    """
    Generate a standardized filename for a financial report.
    
    Parameters:
    - report_type: Type of report (e.g., 'stock_analysis', 'personal_finance', 'document_summary')
    - subject: Subject of the report (e.g., company ticker, person's name, document title)
    
    Returns:
    - Standardized filename with date
    """
    date_str = datetime.now().strftime("%Y%m%d")
    sanitized_subject = "".join(c if c.isalnum() else "_" for c in subject)
    return f"{report_type}_{sanitized_subject}_{date_str}.pdf"

def save_to_json(data: Dict[str, Any], filename: str) -> str:
    """
    Save data to a JSON file.
    
    Parameters:
    - data: Dictionary of data to save
    - filename: Filename to save to
    
    Returns:
    - Path to the saved file
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Data saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {e}")
        return ""

def load_from_json(filename: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Parameters:
    - filename: Filename to load from
    
    Returns:
    - Dictionary of loaded data
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        logger.info(f"Data loaded from {filename}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {e}")
        return {}

def format_financial_report(content: Dict[str, Any], format_type: str = "markdown") -> str:
    """
    Format financial report content into a specified format.
    
    Parameters:
    - content: Report content as a dictionary
    - format_type: Output format (markdown, html, plain)
    
    Returns:
    - Formatted report content
    """
    if format_type == "markdown":
        return _format_markdown_report(content)
    elif format_type == "html":
        return _format_html_report(content)
    else:  # plain text
        return _format_plain_report(content)

def _format_markdown_report(content: Dict[str, Any]) -> str:
    """Format report as markdown"""
    markdown = ""
    
    # Add title
    if "title" in content:
        markdown += f"# {content['title']}\n\n"
    
    # Add date
    if "date" in content:
        markdown += f"*Generated on {content['date']}*\n\n"
    
    # Add summary
    if "summary" in content:
        markdown += f"## Summary\n\n{content['summary']}\n\n"
    
    # Add sections
    if "sections" in content:
        for section in content["sections"]:
            markdown += f"## {section['title']}\n\n"
            markdown += f"{section['content']}\n\n"
    
    # Add tables
    if "tables" in content:
        for table in content["tables"]:
            markdown += f"### {table['title']}\n\n"
            
            # Create header row
            header = "| " + " | ".join(table["headers"]) + " |\n"
            separator = "| " + " | ".join(["---"] * len(table["headers"])) + " |\n"
            
            markdown += header + separator
            
            # Add data rows
            for row in table["data"]:
                markdown += "| " + " | ".join(str(cell) for cell in row) + " |\n"
            
            markdown += "\n"
    
    # Add conclusion
    if "conclusion" in content:
        markdown += f"## Conclusion\n\n{content['conclusion']}\n\n"
    
    # Add disclaimer
    if "disclaimer" in content:
        markdown += f"*{content['disclaimer']}*\n"
    
    return markdown

def _format_html_report(content: Dict[str, Any]) -> str:
    """Format report as HTML"""
    html = "<!DOCTYPE html>\n<html>\n<head>\n"
    html += "<meta charset='UTF-8'>\n"
    html += "<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
    html += "<style>\n"
    html += "body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }\n"
    html += "h1 { color: #2c3e50; }\n"
    html += "h2 { color: #3498db; margin-top: 25px; }\n"
    html += "h3 { color: #2980b9; }\n"
    html += "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n"
    html += "th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }\n"
    html += "th { background-color: #f2f2f2; }\n"
    html += "tr:hover { background-color: #f5f5f5; }\n"
    html += ".disclaimer { font-style: italic; color: #7f8c8d; margin-top: 30px; }\n"
    html += ".date { color: #7f8c8d; font-style: italic; }\n"
    html += "</style>\n"
    
    # Add title
    if "title" in content:
        html += f"<title>{content['title']}</title>\n"
    
    html += "</head>\n<body>\n"
    
    # Add title
    if "title" in content:
        html += f"<h1>{content['title']}</h1>\n"
    
    # Add date
    if "date" in content:
        html += f"<p class='date'>Generated on {content['date']}</p>\n"
    
    # Add summary
    if "summary" in content:
        html += f"<h2>Summary</h2>\n<p>{content['summary']}</p>\n"
    
    # Add sections
    if "sections" in content:
        for section in content["sections"]:
            html += f"<h2>{section['title']}</h2>\n"
            html += f"<p>{section['content']}</p>\n"
    
    # Add tables
    if "tables" in content:
        for table in content["tables"]:
            html += f"<h3>{table['title']}</h3>\n"
            html += "<table>\n<thead>\n<tr>\n"
            
            # Create header row
            for header in table["headers"]:
                html += f"<th>{header}</th>\n"
            
            # finance_agent/utils.py (continued)

            html += "</tr>\n</thead>\n<tbody>\n"
            
            # Add data rows
            for row in table["data"]:
                html += "<tr>\n"
                for cell in row:
                    html += f"<td>{cell}</td>\n"
                html += "</tr>\n"
            
            html += "</tbody>\n</table>\n"
    
    # Add conclusion
    if "conclusion" in content:
        html += f"<h2>Conclusion</h2>\n<p>{content['conclusion']}</p>\n"
    
    # Add disclaimer
    if "disclaimer" in content:
        html += f"<p class='disclaimer'>{content['disclaimer']}</p>\n"
    
    html += "</body>\n</html>"
    return html

def _format_plain_report(content: Dict[str, Any]) -> str:
    """Format report as plain text"""
    plain = ""
    
    # Add title
    if "title" in content:
        plain += f"{content['title'].upper()}\n"
        plain += "=" * len(content['title']) + "\n\n"
    
    # Add date
    if "date" in content:
        plain += f"Generated on {content['date']}\n\n"
    
    # Add summary
    if "summary" in content:
        plain += "SUMMARY\n-------\n\n"
        plain += f"{content['summary']}\n\n"
    
    # Add sections
    if "sections" in content:
        for section in content["sections"]:
            plain += f"{section['title'].upper()}\n"
            plain += "-" * len(section['title']) + "\n\n"
            plain += f"{section['content']}\n\n"
    
    # Add tables
    if "tables" in content:
        for table in content["tables"]:
            plain += f"{table['title']}\n"
            plain += "-" * len(table['title']) + "\n\n"
            
            # Determine column widths
            col_widths = [len(header) for header in table["headers"]]
            for row in table["data"]:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
            
            # Create header row
            header = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(table["headers"]))
            plain += header + "\n"
            plain += "  ".join("-" * w for w in col_widths) + "\n"
            
            # Add data rows
            for row in table["data"]:
                row_str = "  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
                plain += row_str + "\n"
            
            plain += "\n"
    
    # Add conclusion
    if "conclusion" in content:
        plain += "CONCLUSION\n----------\n\n"
        plain += f"{content['conclusion']}\n\n"
    
    # Add disclaimer
    if "disclaimer" in content:
        plain += f"{content['disclaimer']}\n"
    
    return plain

def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Parameters:
    - directory_path: Path to the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory {directory_path}")