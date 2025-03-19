import os
import requests
import json
import openai
from bs4 import BeautifulSoup
from supabase import create_client
from datetime import datetime
import time
import re

class FinanceNewsSystem:
    def __init__(self, openai_api_key, supabase_url, supabase_key):
        # Initialize API keys and clients
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Initialize Supabase client
        self.supabase = create_client(supabase_url, supabase_key)
        
        # News sources configuration
        self.sources = {
            "bloomberg": {
                "url": "https://www.bloomberg.com/markets",
                "selector": "article"  # CSS selector for articles
            },
            "yahoo_finance": {
                "url": "https://finance.yahoo.com/news/",
                "selector": "div.Ov\(h\)"  # CSS selector for Yahoo Finance news items
            }
        }

    def fetch_bloomberg_news(self):
        """Scrape news from Bloomberg Finance"""
        print("Fetching Bloomberg news...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(self.sources["bloomberg"]["url"], headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            for article in soup.select(self.sources["bloomberg"]["selector"]):
                # Extract headline, might need adjustments based on actual Bloomberg structure
                headline_element = article.select_one("h3") or article.select_one("h1")
                if headline_element:
                    headline = headline_element.text.strip()
                    
                    # Extract URL
                    link_element = article.select_one("a")
                    url = ""
                    if link_element and link_element.has_attr('href'):
                        url = link_element['href']
                        if not url.startswith('http'):
                            url = "https://www.bloomberg.com" + url
                    
                    # Extract summary if available
                    summary_element = article.select_one("p")
                    summary = summary_element.text.strip() if summary_element else ""
                    
                    articles.append({
                        "source": "Bloomberg",
                        "headline": headline,
                        "summary": summary,
                        "url": url,
                        "timestamp": datetime.now().isoformat(),
                        "processed": False
                    })
            
            print(f"Found {len(articles)} Bloomberg articles")
            return articles
        
        except Exception as e:
            print(f"Error fetching Bloomberg news: {str(e)}")
            return []

    def fetch_yahoo_finance_news(self):
        """Fetch news from Yahoo Finance"""
        print("Fetching Yahoo Finance news...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(self.sources["yahoo_finance"]["url"], headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            for article in soup.select("li.js-stream-content"):
                # Extract headline
                headline_element = article.select_one("h3")
                if headline_element:
                    headline = headline_element.text.strip()
                    
                    # Extract URL
                    link_element = article.select_one("a")
                    url = ""
                    if link_element and link_element.has_attr('href'):
                        url = link_element['href']
                        if not url.startswith('http'):
                            url = "https://finance.yahoo.com" + url
                    
                    # Extract summary if available
                    summary_element = article.select_one("p")
                    summary = summary_element.text.strip() if summary_element else ""
                    
                    articles.append({
                        "source": "Yahoo Finance",
                        "headline": headline,
                        "summary": summary,
                        "url": url,
                        "timestamp": datetime.now().isoformat(),
                        "processed": False
                    })
            
            print(f"Found {len(articles)} Yahoo Finance articles")
            return articles
        
        except Exception as e:
            print(f"Error fetching Yahoo Finance news: {str(e)}")
            return []

    def summarize_with_gpt4(self, article):
        """Use GPT-4 to summarize an article"""
        try:
            # Combine headline and summary for GPT processing
            content = f"Headline: {article['headline']}\nSummary: {article['summary']}"
            
            # Create the prompt for GPT-4
            prompt = f"""
            Summarize the following financial news article in 2-3 concise sentences. 
            Focus on key financial insights, market trends, and company performance if applicable.
            
            {content}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in concise news summarization."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
        
        except Exception as e:
            print(f"Error summarizing with GPT-4: {str(e)}")
            return article['summary']  # Return original summary if GPT processing fails

    def assess_relevance_with_gpt4(self, article, summary):
        """Use GPT-4 to assess and enhance relevance based on current events"""
        try:
            prompt = f"""
            Assess the relevance of this financial news summary in the context of current market conditions. 
            If relevant, enhance the summary to highlight connections to broader economic trends or events. 
            If not highly relevant, explain why briefly.
            
            Article: {article['headline']}
            Summary: {summary}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst who evaluates news relevance and enhances summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            
            enhanced_content = response.choices[0].message.content.strip()
            
            # Extract relevance score using simple pattern matching
            relevance_pattern = r"relevance:?\s*(\d+)\/10"
            relevance_match = re.search(relevance_pattern, enhanced_content, re.IGNORECASE)
            
            relevance_score = int(relevance_match.group(1)) if relevance_match else 5  # Default to medium relevance
            
            # Clean up the enhanced content by removing any relevance scoring text
            final_content = re.sub(relevance_pattern, "", enhanced_content, flags=re.IGNORECASE).strip()
            
            return {
                "enhanced_summary": final_content,
                "relevance_score": relevance_score
            }
        
        except Exception as e:
            print(f"Error assessing relevance with GPT-4: {str(e)}")
            return {
                "enhanced_summary": summary,
                "relevance_score": 5  # Default to medium relevance
            }

    def store_in_supabase(self, article):
        """Store processed article in Supabase"""
        try:
            result = self.supabase.table('finance_news').insert(article).execute()
            return result
        except Exception as e:
            print(f"Error storing in Supabase: {str(e)}")
            return None

    def process_articles(self, articles):
        """Process a batch of articles through the entire pipeline"""
        processed_articles = []
        
        for article in articles:
            try:
                # Step 1: Summarize with GPT-4
                summary = self.summarize_with_gpt4(article)
                
                # Step 2: Assess and enhance relevance
                relevance_data = self.assess_relevance_with_gpt4(article, summary)
                
                # Create processed article object
                processed_article = {
                    "source": article["source"],
                    "headline": article["headline"],
                    "original_summary": article["summary"],
                    "gpt_summary": summary,
                    "enhanced_summary": relevance_data["enhanced_summary"],
                    "relevance_score": relevance_data["relevance_score"],
                    "url": article["url"],
                    "timestamp": article["timestamp"],
                    "processed_at": datetime.now().isoformat()
                }
                
                # Step 3: Store in Supabase
                self.store_in_supabase(processed_article)
                
                processed_articles.append(processed_article)
                
                # Sleep briefly to manage API rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing article: {str(e)}")
                continue
        
        return processed_articles

    def run_pipeline(self):
        """Run the complete news processing pipeline"""
        # Step 1: Fetch news from sources
        bloomberg_articles = self.fetch_bloomberg_news()
        yahoo_articles = self.fetch_yahoo_finance_news()
        
        # Combine all articles
        all_articles = bloomberg_articles + yahoo_articles
        
        # Step 2: Process all articles
        processed_articles = self.process_articles(all_articles)
        
        print(f"Successfully processed {len(processed_articles)} articles")
        return processed_articles


# Example usage
if __name__ == "__main__":
    # Load environment variables or use configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    # Initialize the system
    news_system = FinanceNewsSystem(openai_api_key, supabase_url, supabase_key)
    
    # Run the pipeline
    news_system.run_pipeline()