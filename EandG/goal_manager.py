import os
from database import DatabaseManager
from nlp_processor import NLPProcessor
from datetime import datetime, timedelta
import pandas as pd

class GoalManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.nlp = NLPProcessor()
    
    def process_goal_input(self, user_input):
        """Process natural language goal input"""
        goal_details = self.nlp.extract_goal_details(user_input)
        
        if not goal_details:
            return {"success": False, "message": "Could not extract goal details. Please try again."}
        
        # Add goal to database
        result = self.db.add_goal(
            target_amount=goal_details["target_amount"],
            purpose=goal_details["purpose"],
            deadline=goal_details["deadline"]
        )
        
        if result:
            return {
                "success": True,
                "message": f"Goal added: Save ${goal_details['target_amount']} for {goal_details['purpose']} by {goal_details['deadline']}",
                "details": goal_details,
                "goal_id": result[0]["id"] if isinstance(result, list) and result else None
            }
        else:
            return {"success": False, "message": "Failed to add goal to database"}
    
    def update_goal_progress(self, goal_id, current_amount):
        """Update the progress of a goal"""
        goal = self.db.get_goal_by_id(goal_id)
        
        if not goal:
            return {"success": False, "message": f"Goal with ID {goal_id} not found"}
        
        result = self.db.update_goal_progress(goal_id, current_amount)
        
        if result:
            progress_percentage = (current_amount / goal["target_amount"]) * 100
            return {
                "success": True,
                "message": f"Goal progress updated: ${current_amount} of ${goal['target_amount']} ({progress_percentage:.1f}%)",
                "progress_percentage": progress_percentage
            }
        else:
            return {"success": False, "message": "Failed to update goal progress"}
    
    def get_active_goals(self):
        """Get all active goals"""
        goals = self.db.get_goals(status="active")
        return goals
    
    def get_all_goals(self):
        """Get all goals regardless of status"""
        return self.db.get_goals()
    
    def get_goal_progress(self, goal_id):
        """Get the progress of a specific goal"""
        goal = self.db.get_goal_by_id(goal_id)
        
        if not goal:
            return {"success": False, "message": f"Goal with ID {goal_id} not found"}
        
        deadline = datetime.fromisoformat(goal["deadline"].replace("Z", "+00:00"))
        now = datetime.now()
        
        # Calculate time remaining
        days_remaining = (deadline - now).days
        start_date = datetime.fromisoformat(goal["created_at"].replace("Z", "+00:00"))
        total_days = (deadline - start_date).days
        
        # Calculate progress percentage
        progress_percentage = (goal["current_amount"] / goal["target_amount"]) * 100
        
        # Calculate time percentage
        time_percentage = ((now - start_date).days / total_days * 100) if total_days > 0 else 100
        
        # Calculate required monthly savings to reach goal
        months_remaining = max(days_remaining / 30, 0.1)  # Avoid division by zero
        required_monthly = (goal["target_amount"] - goal["current_amount"]) / months_remaining
        
        return {
            "success": True,
            "goal": goal,
            "days_remaining": days_remaining,
            "total_days": total_days,
            "months_remaining": months_remaining,
            "progress_percentage": progress_percentage,
            "time_percentage": time_percentage,
            "required_monthly": required_monthly
        }
    
    def calculate_goal_feasibility(self, goal_id, monthly_income=None, monthly_expenses=None):
        """Calculate if a goal is feasible based on current income and expenses"""
        if not monthly_income or not monthly_expenses:
            # Get average monthly income and expenses from last 3 months
            # This would require additional database tables and functions
            # For now, we'll return a placeholder message
            return {"success": False, "message": "Income and expenses data required"}
        
        goal_progress = self.get_goal_progress(goal_id)
        
        if not goal_progress["success"]:
            return goal_progress
        
        monthly_savings = monthly_income - monthly_expenses
        required_monthly = goal_progress["required_monthly"]
        
        feasibility_percentage = (monthly_savings / required_monthly) * 100 if required_monthly > 0 else 100
        
        if feasibility_percentage >= 100:
            feasibility_message = "On track to meet your goal."
        elif feasibility_percentage >= 75:
            feasibility_message = "You're close but may need to increase savings slightly."
        elif feasibility_percentage >= 50:
            feasibility_message = "You need to increase your savings rate to meet this goal."
        else:
            feasibility_message = "This goal may not be realistic with your current savings rate."
        
        return {
            "success": True,
            "feasibility_percentage": feasibility_percentage,
            "feasibility_message": feasibility_message,
            "monthly_savings": monthly_savings,
            "required_monthly": required_monthly
        }
