import os
from openai import OpenAI
from datetime import datetime, timedelta
import json
import re

class NLPProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = "gpt-4-turbo-preview"  # Using GPT-4-mini for cost effectiveness
    
    def extract_expense_details(self, user_input):
        """Extract expense details from natural language input"""
        prompt = f"""
        Extract the following details from this expense description:
        - Amount (in dollars)
        - Category (e.g., groceries, dining, entertainment, utilities, transport)
        - Date (if not specified, assume today)
        - Description (brief summary of the expense)
        
        User input: "{user_input}"
        
        Return the result as a JSON object with the following structure:
        {{
            "amount": float,
            "category": string,
            "date": string (YYYY-MM-DD format),
            "description": string
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Default to today if date is not specified
            if "date" not in result or not result["date"]:
                result["date"] = datetime.now().strftime("%Y-%m-%d")
            
            # Handle relative dates
            if result["date"].lower() in ["today", "now"]:
                result["date"] = datetime.now().strftime("%Y-%m-%d")
            elif result["date"].lower() == "yesterday":
                result["date"] = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            return result
        except Exception as e:
            print(f"Error extracting expense details: {e}")
            return None
    
    def extract_goal_details(self, user_input):
        """Extract goal details from natural language input"""
        prompt = f"""
        Extract the following details from this financial goal description:
        - Target amount (in dollars)
        - Purpose (what the goal is for)
        - Deadline (date by when the goal should be achieved)
        
        User input: "{user_input}"
        
        Return the result as a JSON object with the following structure:
        {{
            "target_amount": float,
            "purpose": string,
            "deadline": string (YYYY-MM-DD format)
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Convert month names to dates if needed
            if "deadline" in result and result["deadline"]:
                # Handle month names like "June" or "June 2023"
                month_pattern = r"^(January|February|March|April|May|June|July|August|September|October|November|December)(\s+\d{4})?$"
                match = re.match(month_pattern, result["deadline"], re.IGNORECASE)
                
                if match:
                    month_name = match.group(1)
                    year = match.group(2).strip() if match.group(2) else str(datetime.now().year)
                    
                    # Convert month name to month number
                    month_dict = {
                        "january": 1, "february": 2, "march": 3, "april": 4,
                        "may": 5, "june": 6, "july": 7, "august": 8,
                        "september": 9, "october": 10, "november": 11, "december": 12
                    }
                    
                    month_num = month_dict[month_name.lower()]
                    result["deadline"] = f"{year}-{month_num:02d}-01"
            
            return result
        except Exception as e:
            print(f"Error extracting goal details: {e}")
            return None
