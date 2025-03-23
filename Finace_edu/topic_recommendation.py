"""
Topic Recommendation Module for Personal Finance Teaching Assistant

This module contains functions to recommend the next learning topic
based on user progress and knowledge level.
"""

from typing import Dict, List, Any

def recommend_next_topic(user_id: str) -> Dict[str, Any]:
    """Recommend the next topic based on user progress."""
    
    # Placeholder logic to determine user progress
    # In a real implementation, you would fetch user progress from a database
    user_progress = get_user_progress(user_id)  # Assume this function fetches user progress
    
    completed_modules = user_progress.get("completed_modules", [])
    knowledge_level = user_progress.get("knowledge_level", 1)
    
    # Define a list of topics based on knowledge level
    topics = {
        1: ["Introduction to Personal Finance", "Understanding Money", "Basic Budgeting"],
        2: ["Intermediate Budgeting", "Saving Strategies", "Debt Management"],
        3: ["Investing Basics", "Retirement Planning", "Insurance Fundamentals"],
        4: ["Advanced Investing", "Tax Strategies", "Estate Planning"],
        5: ["Financial Independence", "Wealth Building Strategies", "Philanthropy and Giving Back"]
    }
    
    # Determine the next topic based on the user's knowledge level and completed modules
    next_topics = topics.get(knowledge_level, [])
    
    # Filter out topics that the user has already completed
    next_topics = [topic for topic in next_topics if topic not in completed_modules]
    
    # If no topics are left, suggest a review or advanced topic
    if not next_topics:
        next_topics = ["Review Previous Topics", "Advanced Financial Strategies"]
    
    return {
        "title": "Next Topic Recommendation",
        "topics": next_topics,
        "reason": "This topic is recommended based on your current knowledge level."
    }

def get_user_progress(user_id: str) -> Dict[str, Any]:
    """Mock function to simulate fetching user progress."""
    # In a real implementation, this function would query a database
    return {
        "completed_modules": ["Understanding Money", "Basic Budgeting"],
        "knowledge_level": 2
    }