import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Import local modules
from database import DatabaseManager
from expense_tracker import ExpenseTracker
from goal_manager import GoalManager
from nlp_processor import NLPProcessor
from ai_insights import AIInsightGenerator
from visualization import FinanceVisualizer
from utils import UIHelper, DateHelper, ColorPalette

# Initialize services
db = DatabaseManager()
expense_tracker = ExpenseTracker()
goal_manager = GoalManager()
nlp_processor = NLPProcessor()
ai_insight_generator = AIInsightGenerator()
visualizer = FinanceVisualizer()

# Set page config
st.set_page_config(
    page_title="Personal Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px !important;
        font-weight: bold;
        margin-bottom: 20px;
        color: #3498db;
    }
    .card {
        border-radius: 10px;
        background-color: #f8f9fa;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #333;
    }
    .chat-message {
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e6f3ff;
        text-align: right;
    }
    .bot-message {
        background-color: #f0f0f0;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #2980b9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .progress-container {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = "dashboard"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar navigation
st.sidebar.markdown("# 💰 Finance Assistant")
selected_page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Expense Tracking", "Goal Tracking", "Insights", "Settings"],
    format_func=lambda x: f"📊 {x}" if x == "Dashboard" else 
                         f"💸 {x}" if x == "Expense Tracking" else 
                         f"🎯 {x}" if x == "Goal Tracking" else 
                         f"💡 {x}" if x == "Insights" else 
                         f"⚙️ {x}"
)

st.session_state.page = selected_page.lower()

# Dashboard Page
if st.session_state.page == "dashboard":
    st.markdown('<h1 class="main-header">Financial Dashboard</h1>', unsafe_allow_html=True)
    
    # Get current date
    now = datetime.now()
    current_month = now.strftime("%B %Y")
    
    # Get expense data
    monthly_data = expense_tracker.get_monthly_summary()
    spending_trends = expense_tracker.get_spending_trends(months=6)
    active_goals = goal_manager.get_active_goals()
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_spent = monthly_data.get("total", 0) if monthly_data and "total" in monthly_data else 0
        mom_change = monthly_data.get("month_over_month", 0) if monthly_data and "month_over_month" in monthly_data else 0
        UIHelper.create_metric_card(
            "Total Spent This Month",
            UIHelper.format_currency(total_spent),
            mom_change
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        active_goal_count = len(active_goals) if active_goals else 0
        UIHelper.create_metric_card(
            "Active Goals",
            str(active_goal_count),
            help_text="Financial goals you're currently tracking"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        categories = monthly_data.get("categories", {}) if monthly_data and "categories" in monthly_data else {}
        top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "None"
        top_category_amount = max(categories.values()) if categories else 0
        
        UIHelper.create_metric_card(
            "Top Spending Category",
            top_category.title(),
            help_text=f"You spent {UIHelper.format_currency(top_category_amount)} in this category"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Expense trend chart
    st.markdown('<h2 class="section-header">Spending Trends</h2>', unsafe_allow_html=True)
    
    if spending_trends and "monthly_totals" in spending_trends:
        trend_chart = visualizer.create_monthly_trend_chart(spending_trends["monthly_totals"])
        st.plotly_chart(trend_chart, use_container_width=True)
    else:
        st.info("Not enough data to display spending trends. Start tracking your expenses to see trends over time.")
    
    # Category breakdown
    st.markdown('<h2 class="section-header">Expense Breakdown</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        if categories:
            pie_chart = visualizer.create_category_pie_chart(categories)
            st.plotly_chart(pie_chart, use_container_width=True)
        else:
            st.info("No expense data available for this month.")
    
    with col2:
        if spending_trends and "category_trends" in spending_trends:
            top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5] if categories else []
            top_category_names = [cat[0] for cat in top_categories]
            
            category_chart = visualizer.create_category_trend_chart(
                spending_trends["category_trends"], 
                categories=top_category_names
            )
            st.plotly_chart(category_chart, use_container_width=True)
        else:
            st.info("Not enough data to display category trends.")
    
    # Goal progress
    if active_goals:
        st.markdown('<h2 class="section-header">Goal Progress</h2>', unsafe_allow_html=True)
        
        for goal in active_goals[:3]:  # Show top 3 goals
            progress = goal_manager.get_goal_progress(goal["id"])
            
            if progress["success"]:
                goal_progress = progress["progress_percentage"]
                st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                UIHelper.create_progress_bar(
                    f"{goal['purpose']} - Target: {goal['deadline']}",
                    goal["current_amount"],
                    goal["target_amount"],
                    goal_progress
                )
                st.markdown('</div>',