import os
from database import DatabaseManager
from nlp_processor import NLPProcessor
from datetime import datetime, timedelta
import pandas as pd

class ExpenseTracker:
    def __init__(self):
        self.db = DatabaseManager()
        self.nlp = NLPProcessor()
    
    def process_expense_input(self, user_input):
        """Process natural language expense input"""
        expense_details = self.nlp.extract_expense_details(user_input)
        
        if not expense_details:
            return {"success": False, "message": "Could not extract expense details. Please try again."}
        
        # Add expense to database
        result = self.db.add_expense(
            amount=expense_details["amount"],
            category=expense_details["category"],
            date=expense_details["date"],
            description=expense_details["description"]
        )
        
        if result:
            return {
                "success": True,
                "message": f"Expense added: ${expense_details['amount']} for {expense_details['description']} in category {expense_details['category']}",
                "details": expense_details
            }
        else:
            return {"success": False, "message": "Failed to add expense to database"}
    
    def get_monthly_summary(self, year=None, month=None):
        """Get a summary of expenses for the current month"""
        if year is None or month is None:
            now = datetime.now()
            year = now.year
            month = now.month
        
        month_name = datetime(year, month, 1).strftime("%B")
        expenses = self.db.get_monthly_expenses(year, month)
        
        if not expenses:
            return {
                "success": True,
                "message": f"No expenses found for {month_name} {year}",
                "total": 0,
                "categories": {},
                "expenses": []
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(expenses)
        
        # Calculate total and category breakdown
        total_spent = df["amount"].sum()
        category_totals = df.groupby("category")["amount"].sum().to_dict()
        
        # Calculate percentage of total for each category
        category_percentages = {
            category: (amount / total_spent) * 100 
            for category, amount in category_totals.items()
        }
        
        # Get previous month for comparison
        if month == 1:
            prev_month = 12
            prev_year = year - 1
        else:
            prev_month = month - 1
            prev_year = year
        
        prev_expenses = self.db.get_monthly_expenses(prev_year, prev_month)
        prev_total = 0
        prev_categories = {}
        
        if prev_expenses:
            prev_df = pd.DataFrame(prev_expenses)
            prev_total = prev_df["amount"].sum()
            prev_categories = prev_df.groupby("category")["amount"].sum().to_dict()
        
        # Calculate month-over-month changes
        mom_change = ((total_spent - prev_total) / prev_total * 100) if prev_total > 0 else 0
        
        category_changes = {}
        for category, amount in category_totals.items():
            prev_amount = prev_categories.get(category, 0)
            change = ((amount - prev_amount) / prev_amount * 100) if prev_amount > 0 else 0
            category_changes[category] = change
        
        return {
            "success": True,
            "message": f"Monthly summary for {month_name} {year}",
            "total": total_spent,
            "categories": category_totals,
            "percentages": category_percentages,
            "month_over_month": mom_change,
            "category_changes": category_changes,
            "expenses": expenses
        }
    
    def get_spending_trends(self, months=6):
        """Get spending trends for the last N months"""
        now = datetime.now()
        
        # Generate list of months to analyze
        month_list = []
        for i in range(months):
            month = now.month - i
            year = now.year
            
            while month <= 0:
                month += 12
                year -= 1
                
            month_list.append((year, month))
        
        # Reverse to get chronological order
        month_list.reverse()
        
        # Get expenses for each month
        monthly_totals = []
        monthly_categories = {}
        
        for year, month in month_list:
            expenses = self.db.get_monthly_expenses(year, month)
            month_name = datetime(year, month, 1).strftime("%b %Y")
            
            if expenses:
                df = pd.DataFrame(expenses)
                total = df["amount"].sum()
                categories = df.groupby("category")["amount"].sum().to_dict()
                
                monthly_totals.append({"month": month_name, "total": total})
                
                # Track category spending over time
                for category, amount in categories.items():
                    if category not in monthly_categories:
                        monthly_categories[category] = []
                    
                    monthly_categories[category].append({
                        "month": month_name,
                        "amount": amount
                    })
            else:
                monthly_totals.append({"month": month_name, "total": 0})
        
        return {
            "success": True,
            "monthly_totals": monthly_totals,
            "category_trends": monthly_categories
        }

    def get_expenses_by_month(self, year, month):
        """Get expenses for a specific month"""
        return self.db.get_monthly_expenses(year, month)
        
    def get_analytics(self, start_date, end_date):
        """Get analytics for a date range"""
        expenses = self.db.get_expenses(start_date=start_date, end_date=end_date)
        
        if not expenses:
            return {
                "success": True,
                "message": "No expenses found for the selected period",
                "total_spent": 0,
                "average_monthly": 0,
                "highest_month": {"month": None, "amount": 0},
                "monthly_trend": [],
                "category_breakdown": {}
            }
        
        df = pd.DataFrame(expenses)
        
        # Calculate total spent
        total_spent = df["amount"].sum()
        
        # Group by month and calculate monthly totals
        df["month"] = pd.to_datetime(df["date"]).dt.strftime("%b %Y")
        monthly_totals = df.groupby("month")["amount"].sum()
        
        # Calculate average monthly spending
        num_months = len(monthly_totals)
        average_monthly = total_spent / num_months if num_months > 0 else 0
        
        # Find highest spending month
        highest_month = {
            "month": monthly_totals.idxmax() if not monthly_totals.empty else None,
            "amount": monthly_totals.max() if not monthly_totals.empty else 0
        }
        
        # Create monthly trend data
        monthly_trend = [
            {"month": month, "total": amount}
            for month, amount in monthly_totals.items()
        ]
        
        # Calculate category breakdown
        category_breakdown = df.groupby("category")["amount"].sum().to_dict()
        
        return {
            "success": True,
            "total_spent": total_spent,
            "average_monthly": average_monthly,
            "highest_month": highest_month,
            "monthly_trend": monthly_trend,
            "category_breakdown": category_breakdown
        }
