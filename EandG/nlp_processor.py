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
        Please extract the following details from the expense description and return the result as a JSON object:
        - Amount (in dollars)
        - Category (e.g., groceries, dining, entertainment)
        - Date (format: YYYY-MM-DD)
        - Description (a brief summary of the expense)
        
        Example: "I spent $50 on groceries yesterday."
        
        User input: "{user_input}"
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            # Debugging: Print the raw response
            print(f"Raw response: {response}")  # Add this line
            
            # Check if the response is valid
            if not response.choices or not response.choices[0].message.content:
                print("No valid response from the model.")
                return None
            
            result = json.loads(response.choices[0].message.content)
            
            # Debugging: Print the extracted result
            print(f"Extracted result: {result}")  # Add this line
            
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

# Test code
if __name__ == "__main__":
    # Create an instance of NLPProcessor
    nlp = NLPProcessor()
    
    # Test inputs
    test_inputs = [
        "I spent $50 on groceries yesterday.",
        "Bought a new shirt for $30 on 2023-10-01.",
        "Dinner cost me $45 last night.",
        "I paid $100 for utilities this month."
    ]

    # Test expense extraction
    print("\nTesting expense extraction:")
    print("-" * 50)
    for input_str in test_inputs:
        result = nlp.extract_expense_details(input_str)
        print(f"\nInput: {input_str}")
        print(f"Result: {result}")
    
    # Test goal extraction
    print("\nTesting goal extraction:")
    print("-" * 50)
    goal_inputs = [
        "I want to save $5000 for a vacation by December 2024",
        "Need to save $10000 for a car down payment by June",
        "Save $2000 for emergency fund by the end of this year"
    ]
    
    for input_str in goal_inputs:
        result = nlp.extract_goal_details(input_str)
        print(f"\nInput: {input_str}")
        print(f"Result: {result}")