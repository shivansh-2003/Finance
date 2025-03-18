import os
from database import DatabaseManager
from nlp_processor import NLPProcessor
from datetime import datetime, timedelta, timezone
import pandas as pd
from dateutil import parser  # Import the parser from dateutil

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
            return {"success": False, "message": "Goal not found."}
        
        # Debugging: Print the raw deadline string
        print(f"Raw Deadline String: {goal['deadline']}")
        
        # Use dateutil.parser to parse the deadline
        try:
            deadline = parser.isoparse(goal["deadline"])  # This will handle the timezone correctly
        except Exception as e:
            print(f"Error parsing deadline: {e}")
            raise ValueError("Failed to parse deadline.")

        # Debugging: Print the parsed deadline
        print(f"Parsed Deadline: {deadline}, Timezone: {deadline.tzinfo}")

        # If the deadline is still naive, force it to be UTC
        if deadline.tzinfo is None:
            print("Deadline is offset-naive, setting to UTC.")
            deadline = deadline.replace(tzinfo=timezone.utc)

        # Get the current time as an offset-aware datetime
        start_date = datetime.now(timezone.utc)  # Set timezone to UTC directly

        # Debugging: Print the start date
        print(f"Start Date: {start_date}, Timezone: {start_date.tzinfo}")

        # Calculate days remaining
        days_remaining = (deadline - start_date).days

        # Calculate total days for the goal duration
        total_days = (deadline - parser.isoparse(goal["created_at"])).days  # Assuming goal["created_at"] is in ISO format

        # Calculate time percentage
        time_percentage = ((total_days - days_remaining) / total_days * 100) if total_days > 0 else 0

        # Calculate required monthly savings
        if total_days > 0:
            remaining_amount = goal["target_amount"] - goal["current_amount"]
            months_remaining = (days_remaining // 30) + (1 if days_remaining % 30 > 0 else 0)  # Round up to the next month
            required_monthly = remaining_amount / months_remaining if months_remaining > 0 else 0
        else:
            required_monthly = 0

        # Prepare the progress dictionary
        progress = {
            "success": True,
            "progress_percentage": (goal["current_amount"] / goal["target_amount"]) * 100 if goal["target_amount"] > 0 else 0,
            "days_remaining": days_remaining,
            "time_percentage": time_percentage,  # Include time percentage
            "total_days": total_days,  # Include total days
            "required_monthly": required_monthly,  # Include required monthly savings
            "goal": goal  # Include the goal details if needed
        }

        return progress
    
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
