import os
import json
from datetime import datetime, timedelta
from database import DatabaseManager
from anthropic import Anthropic
import pandas as pd

class AIInsightGenerator:
    def __init__(self):
        self.db = DatabaseManager()
        self.anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = "claude-3-sonnet-20240229"  # Using more advanced model for complex insights
    
    def generate_monthly_summary(self, year=None, month=None):
        """Generate conversational monthly summary with insights"""
        if year is None or month is None:
            now = datetime.now()
            year = now.year
            month = now.month
        
        month_name = datetime(year, month, 1).strftime("%B %Y")
        expenses = self.db.get_monthly_expenses(year, month)
        
        if not expenses:
            return {
                "success": True,
                "message": f"No expenses found for {month_name}. Start tracking your expenses to get insights.",
                "summary": f"No expenses found for {month_name}."
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(expenses)
        total_spent = df["amount"].sum()
        category_totals = df.groupby("category")["amount"].sum().to_dict()
        
        # Get previous month for comparison
        if month == 1:
            prev_month = 12
            prev_year = year - 1
        else:
            prev_month = month - 1
            prev_year = year
        
        prev_expenses = self.db.get_monthly_expenses(prev_year, prev_month)
        prev_month_name = datetime(prev_year, prev_month, 1).strftime("%B %Y")
        
        prev_total = 0
        prev_categories = {}
        
        if prev_expenses:
            prev_df = pd.DataFrame(prev_expenses)
            prev_total = prev_df["amount"].sum()
            prev_categories = prev_df.groupby("category")["amount"].sum().to_dict()
        
        # Prepare data for Claude
        expense_data = {
            "current_month": month_name,
            "previous_month": prev_month_name,
            "total_spent": total_spent,
            "previous_total": prev_total,
            "categories": {
                category: {
                    "amount": amount,
                    "previous_amount": prev_categories.get(category, 0),
                    "percent_of_total": (amount / total_spent) * 100
                }
                for category, amount in category_totals.items()
            }
        }
        
        # Generate insights with Claude
        prompt = f"""
        As a personal finance assistant, analyze the following monthly expense data and provide insights in a conversational tone.
        Highlight notable changes, patterns, and suggest actionable advice for the user.
        
        Expense data: {json.dumps(expense_data, indent=2)}
        
        Your response should include:
        1. A summary of overall spending for {month_name}
        2. Month-over-month comparison with {prev_month_name}
        3. Analysis of spending by category
        4. At least 2-3 specific, actionable insights or suggestions
        5. A positive, encouraging tone
        
        Keep your response concise (200-300 words) and conversational.
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            summary = response.content[0].text
            
            return {
                "success": True,
                "message": "Generated monthly summary with AI insights",
                "summary": summary,
                "data": expense_data
            }
        except Exception as e:
            print(f"Error generating AI insights: {e}")
            return {
                "success": False, 
                "message": f"Could not generate AI insights. Error: {str(e)}",
                "data": expense_data
            }
    
    def generate_goal_suggestions(self, goal_id, monthly_expenses=None):
        """Generate suggestions to help achieve financial goals"""
        goal = self.db.get_goal_by_id(goal_id)
        
        if not goal:
            return {"success": False, "message": f"Goal with ID {goal_id} not found"}
        
        # Get recent expenses for the last 3 months
        now = datetime.now()
        three_months_ago = now - timedelta(days=90)
        
        if not monthly_expenses:
            recent_expenses = self.db.get_expenses(
                start_date=three_months_ago.strftime("%Y-%m-%d"),
                end_date=now.strftime("%Y-%m-%d")
            )
            
            if not recent_expenses:
                return {
                    "success": False,
                    "message": "Not enough expense data to generate meaningful suggestions"
                }
                
            expenses_df = pd.DataFrame(recent_expenses)
            category_totals = expenses_df.groupby("category")["amount"].sum().to_dict()
            monthly_expenses = {category: amount / 3 for category, amount in category_totals.items()}
        
        # Calculate current savings rate and required savings
        total_monthly_expenses = sum(monthly_expenses.values())
        deadline = datetime.fromisoformat(goal["deadline"].replace("Z", "+00:00"))
        months_remaining = max((deadline - now).days / 30, 0.1)
        amount_needed = goal["target_amount"] - goal["current_amount"]
        monthly_target = amount_needed / months_remaining
        
        # Prepare data for Claude
        goal_data = {
            "goal_purpose": goal["purpose"],
            "target_amount": goal["target_amount"],
            "current_amount": goal["current_amount"],
            "deadline": goal["deadline"],
            "months_remaining": months_remaining,
            "monthly_target": monthly_target,
            "monthly_expenses": monthly_expenses,
            "total_monthly_expenses": total_monthly_expenses
        }
        
        # Generate insights with Claude
        prompt = f"""
        As a personal finance advisor, analyze the following financial goal and expense data.
        Provide personalized suggestions to help the user achieve their savings goal.
        
        Goal and expense data: {json.dumps(goal_data, indent=2)}
        
        Your response should include:
        1. A brief assessment of the goal's feasibility
        2. 3-5 specific, actionable suggestions to help achieve the goal (e.g., "Reduce dining expenses by $50/week")
        3. Potential categories where spending could be reduced
        4. Alternative strategies like increasing income or extending deadline if appropriate
        5. A positive, encouraging tone
        
        Keep your response concise (250-350 words) and conversational.
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            suggestions = response.content[0].text
            
            return {
                "success": True,
                "message": "Generated goal suggestions with AI",
                "suggestions": suggestions,
                "data": goal_data
            }
        except Exception as e:
            print(f"Error generating goal suggestions: {e}")
            return {
                "success": False, 
                "message": f"Could not generate goal suggestions. Error: {str(e)}",
                "data": goal_data
            }
