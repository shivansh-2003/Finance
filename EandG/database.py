import os
from supabase import create_client, Client
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

load_dotenv()

class DatabaseManager:
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        self.supabase: Client = create_client(url, key)
        self._initialize_tables()
    
    def _initialize_tables(self):
        """Initialize tables if they don't exist (this is simplified - in production, use migrations)"""
        # This is a simplified approach. In production, use proper migrations.
        # These operations assume the tables are already created in Supabase
        pass

    def add_expense(self, amount, category, date, description):
        """Add an expense to the database"""
        data = {
            "amount": amount,
            "category": category,
            "date": date,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        
        response = self.supabase.table("expenses").insert(data).execute()
        return response.data
    
    def add_goal(self, target_amount, purpose, deadline, created_at=None):
        """Add a financial goal to the database"""
        if created_at is None:
            created_at = datetime.now().isoformat()
            
        data = {
            "target_amount": target_amount,
            "current_amount": 0,
            "purpose": purpose,
            "deadline": deadline,
            "created_at": created_at,
            "status": "active"
        }
        
        response = self.supabase.table("goals").insert(data).execute()
        return response.data
    
    def update_goal_progress(self, goal_id, current_amount):
        """Update the progress of a goal"""
        data = {"current_amount": current_amount}
        response = self.supabase.table("goals").update(data).eq("id", goal_id).execute()
        return response.data
    
    def get_expenses(self, start_date=None, end_date=None, category=None):
        """Get expenses with optional filtering"""
        query = self.supabase.table("expenses").select("*")
        
        if start_date:
            query = query.gte("date", start_date)
        if end_date:
            query = query.lte("date", end_date)
        if category:
            query = query.eq("category", category)
            
        # Fix: Use order() correctly for the Supabase Python client
        response = query.execute()
        
        # Sort results by date in Python after fetching
        data = response.data
        if data:
            data.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        return data
    
    def get_expenses_by_category(self, start_date=None, end_date=None):
        """Get expenses grouped by category"""
        expenses = self.get_expenses(start_date, end_date)
        df = pd.DataFrame(expenses)
        
        if df.empty:
            return {}
            
        # Group by category and sum amounts
        category_totals = df.groupby("category")["amount"].sum().to_dict()
        return category_totals
    
    def get_monthly_expenses(self, year=None, month=None):
        """Get expenses for a specific month"""
        if year is None or month is None:
            now = datetime.now()
            year = now.year
            month = now.month
            
        start_date = f"{year}-{month:02d}-01"
        
        # Calculate end date
        if month == 12:
            end_date = f"{year+1}-01-01"
        else:
            end_date = f"{year}-{month+1:02d}-01"
            
        return self.get_expenses(start_date, end_date)
    
    def get_goals(self, status=None):
        """Get all goals or filter by status"""
        query = self.supabase.table("goals").select("*")
        
        if status:
            query = query.eq("status", status)
            
        # Fix: Use execute() directly rather than order()
        response = query.execute()
        
        # Sort results by created_at in Python after fetching
        data = response.data
        if data:
            data.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return data
    
    def get_goal_by_id(self, goal_id):
        """Get a specific goal by ID"""
        response = self.supabase.table("goals").select("*").eq("id", goal_id).execute()
        if response.data:
            return response.data[0]
        return None