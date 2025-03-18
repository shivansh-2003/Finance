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
                st.markdown('</div>', unsafe_allow_html=True)
    
    # AI Insights
    if monthly_data:
        st.markdown('<h2 class="section-header">AI Insights</h2>', unsafe_allow_html=True)
        
        with st.spinner("Generating insights..."):
            current_year, current_month = now.year, now.month
            insights = ai_insight_generator.generate_monthly_summary(current_year, current_month)
            
            if insights["success"]:
                UIHelper.display_insight_card(
                    f"Monthly Summary - {current_month}",
                    insights["summary"]
                )
            else:
                st.info("Could not generate insights. Track more expenses to get personalized insights.")

# Expense Tracking Page
elif st.session_state.page == "expense tracking":
    st.markdown('<h1 class="main-header">Expense Tracker</h1>', unsafe_allow_html=True)
    
    # Create tabs for different functions
    tabs = st.tabs(["Add Expense", "View Expenses", "Analytics"])
    
    # Add Expense Tab
    with tabs[0]:
        st.markdown('<h2 class="section-header">Track New Expense</h2>', unsafe_allow_html=True)
        
        # Chat interface for natural language expense input
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Chat with AI Assistant")
        st.markdown("Describe your expense in natural language, e.g., *\"I spent $50 on groceries yesterday\"*")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message"><p>{message["content"]}</p></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message bot-message"><p>{message["content"]}</p></div>',
                    unsafe_allow_html=True
                )
        
        # Input for new message
        user_input = st.text_input("Enter your expense:", key="expense_input")
        
        if st.button("Submit", key="submit_expense"):
            if user_input:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Process with NLP
                expense_details = nlp_processor.extract_expense_details(user_input)
                
                if expense_details:
                    # Add expense to database
                    result = expense_tracker.add_expense(expense_details)
                    
                    if result["success"]:
                        response = f"Got it! I've recorded ${expense_details['amount']:.2f} for {expense_details['description']} in the {expense_details['category']} category on {expense_details['date']}."
                    else:
                        response = f"Sorry, I couldn't add that expense. Error: {result['message']}"
                else:
                    response = "Sorry, I couldn't understand the expense details. Please try again with a clearer description."
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Clear input field
                st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Alternative form input
        st.markdown('<h3 class="section-header">Manual Entry</h3>', unsafe_allow_html=True)
        
        with st.form("expense_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input("Amount ($)", min_value=0.01, step=1.0)
                date = st.date_input("Date", value=datetime.now())
            
            with col2:
                category_options = [
                    "Groceries", "Dining", "Entertainment", "Utilities", 
                    "Transport", "Housing", "Healthcare", "Shopping", 
                    "Education", "Travel", "Subscriptions", "Personal", "Other"
                ]
                category = st.selectbox("Category", options=category_options)
                description = st.text_input("Description")
            
            submit_form = st.form_submit_button("Add Expense")
        
        if submit_form:
            # Format data
            expense_data = {
                "amount": amount,
                "category": category.lower(),
                "date": date.strftime("%Y-%m-%d"),
                "description": description
            }
            
            # Add to database
            result = expense_tracker.add_expense(expense_data)
            
            if result["success"]:
                st.success(f"Added {UIHelper.format_currency(amount)} expense for {category}")
            else:
                st.error(f"Could not add expense: {result['message']}")
    
    # View Expenses Tab
    with tabs[1]:
        st.markdown('<h2 class="section-header">Expense History</h2>', unsafe_allow_html=True)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            month_options = DateHelper.get_month_options()
            selected_month = st.selectbox(
                "Select Month", 
                options=[opt["value"] for opt in month_options],
                format_func=lambda x: next((opt["label"] for opt in month_options if opt["value"] == x), x)
            )
            year, month = DateHelper.parse_month_value(selected_month)
        
        with col2:
            category_filter = st.multiselect(
                "Filter by Category",
                options=["All"] + [
                    "Groceries", "Dining", "Entertainment", "Utilities", 
                    "Transport", "Housing", "Healthcare", "Shopping", 
                    "Education", "Travel", "Subscriptions", "Personal", "Other"
                ],
                default=["All"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                options=["Date (Latest First)", "Date (Oldest First)", "Amount (High to Low)", "Amount (Low to High)", "Category"]
            )
        
        # Get expenses based on filters
        expenses = expense_tracker.get_expenses_by_month(year, month)
        
        if not expenses:
            st.info(f"No expenses found for {datetime(year, month, 1).strftime('%B %Y')}.")
        else:
            # Apply category filter
            if category_filter and "All" not in category_filter:
                expenses = [e for e in expenses if e["category"].title() in category_filter]
            
            # Apply sorting
            if sort_by == "Date (Latest First)":
                expenses = sorted(expenses, key=lambda x: x["date"], reverse=True)
            elif sort_by == "Date (Oldest First)":
                expenses = sorted(expenses, key=lambda x: x["date"])
            elif sort_by == "Amount (High to Low)":
                expenses = sorted(expenses, key=lambda x: x["amount"], reverse=True)
            elif sort_by == "Amount (Low to High)":
                expenses = sorted(expenses, key=lambda x: x["amount"])
            elif sort_by == "Category":
                expenses = sorted(expenses, key=lambda x: x["category"])
            
            # Display as table
            df = pd.DataFrame(expenses)
            
            # Format columns
            if not df.empty:
                df["amount"] = df["amount"].apply(lambda x: f"${x:.2f}")
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%b %d, %Y")
                df["category"] = df["category"].str.title()
                
                # Rename columns
                df = df.rename(columns={
                    "date": "Date",
                    "description": "Description",
                    "category": "Category",
                    "amount": "Amount"
                })
                
                # Reorder columns
                df = df[["Date", "Description", "Category", "Amount"]]
                
                st.dataframe(df, use_container_width=True)
                
                # Summary
                total = sum(e["amount"] for e in expenses)
                st.markdown(f"**Total:** {UIHelper.format_currency(total)}")
                
                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Export as CSV"):
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"expenses_{year}_{month}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("Export as Excel"):
                        # This would need additional logic to create Excel file
                        st.info("Excel export functionality will be added soon.")
    
    # Analytics Tab
    with tabs[2]:
        st.markdown('<h2 class="section-header">Expense Analytics</h2>', unsafe_allow_html=True)
        
        # Time range selection
        col1, col2 = st.columns(2)
        
        with col1:
            range_options = ["Last Month", "Last 3 Months", "Last 6 Months", "Last Year", "Custom Range"]
            selected_range = st.selectbox("Time Range", options=range_options)
        
        with col2:
            if selected_range == "Custom Range":
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
                end_date = st.date_input("End Date", value=datetime.now())
            else:
                # Calculate date range based on selection
                now = datetime.now()
                if selected_range == "Last Month":
                    start_date = (now - timedelta(days=30)).date()
                elif selected_range == "Last 3 Months":
                    start_date = (now - timedelta(days=90)).date()
                elif selected_range == "Last 6 Months":
                    start_date = (now - timedelta(days=180)).date()
                else:  # Last Year
                    start_date = (now - timedelta(days=365)).date()
                
                end_date = now.date()
        
        # Get analytics
        if selected_range == "Custom Range":
            analytics = expense_tracker.get_analytics(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        else:
            analytics = expense_tracker.get_analytics(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if not analytics or "total_spent" not in analytics:
            st.info("Not enough expense data to display analytics.")
        else:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                UIHelper.create_metric_card(
                    "Total Spent",
                    UIHelper.format_currency(analytics["total_spent"])
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                UIHelper.create_metric_card(
                    "Average Monthly",
                    UIHelper.format_currency(analytics["average_monthly"])
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                UIHelper.create_metric_card(
                    "Highest Month",
                    UIHelper.format_currency(analytics["highest_month"]["amount"]),
                    help_text=f"{analytics['highest_month']['month']}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Trend chart
            if "monthly_trend" in analytics and analytics["monthly_trend"]:
                trend_chart = visualizer.create_monthly_trend_chart(analytics["monthly_trend"])
                st.plotly_chart(trend_chart, use_container_width=True)
            
            # Category breakdown
            if "category_breakdown" in analytics and analytics["category_breakdown"]:
                col1, col2 = st.columns(2)
                
                with col1:
                    pie_chart = visualizer.create_category_pie_chart(analytics["category_breakdown"])
                    st.plotly_chart(pie_chart, use_container_width=True)
                
                with col2:
                    # Format for bar chart
                    categories = list(analytics["category_breakdown"].keys())
                    values = list(analytics["category_breakdown"].values())
                    
                    df = pd.DataFrame({"category": categories, "amount": values})
                    df = df.sort_values("amount", ascending=False)
                    
                    bar_chart = px.bar(
                        df,
                        x="category",
                        y="amount",
                        title="Spending by Category",
                        labels={"category": "Category", "amount": "Amount ($)"}
                    )
                    
                    st.plotly_chart(bar_chart, use_container_width=True)

# Goal Tracking Page
elif st.session_state.page == "goal tracking":
    st.markdown('<h1 class="main-header">Financial Goals</h1>', unsafe_allow_html=True)
    
    # Create tabs for different functions
    tabs = st.tabs(["My Goals", "Set New Goal", "Goal Insights"])
    
    # My Goals Tab
    with tabs[0]:
        st.markdown('<h2 class="section-header">Your Financial Goals</h2>', unsafe_allow_html=True)
        
        # Get all goals
        goals = goal_manager.get_all_goals()
        
        if not goals:
            st.info("You haven't set any financial goals yet. Use the 'Set New Goal' tab to create your first goal!")
        else:
            # Display active goals first
            active_goals = [g for g in goals if not g.get("completed", False)]
            completed_goals = [g for g in goals if g.get("completed", False)]
            
            # Active goals section
            if active_goals:
                st.markdown('<h3 class="section-header">Active Goals</h3>', unsafe_allow_html=True)
                
                for goal in active_goals:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Get goal progress
                        progress = goal_manager.get_goal_progress(goal["id"])
                        
                        if progress["success"]:
                            goal_progress = progress["progress_percentage"]
                            time_progress = progress["time_percentage"]
                            
                            st.markdown(f'<div class="card">', unsafe_allow_html=True)
                            st.markdown(f"### {goal['purpose']}")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown(f"**Target:** {UIHelper.format_currency(goal['target_amount'])}")
                                st.markdown(f"**Current:** {UIHelper.format_currency(goal['current_amount'])}")
                            
                            with col_b:
                                deadline = datetime.fromisoformat(goal["deadline"].replace("Z", "+00:00")).strftime("%b %d, %Y")
                                st.markdown(f"**Deadline:** {deadline}")
                                days_left = progress["days_remaining"]
                                st.markdown(f"**Days Left:** {days_left}")
                            
                            # Progress bar
                            UIHelper.create_progress_bar(
                                "Savings Progress",
                                goal["current_amount"],
                                goal["target_amount"],
                                goal_progress
                            )
                            
                            # Time progress
                            color = "#e74c3c" if time_progress > goal_progress else "#2ecc71"
                            UIHelper.create_progress_bar(
                                "Time Progress",
                                days_left,
                                progress["total_days"],
                                time_progress,
                                color=color
                            )
                            
                            # Status message
                            if goal_progress >= 100:
                                st.success("🎉 Goal achieved! Congratulations!")
                            elif time_progress > goal_progress:
                                st.warning(f"You're behind schedule. You should have saved {UIHelper.format_currency(goal['target_amount'] * time_progress / 100)} by now.")
                            else:
                                st.success("You're ahead of schedule!")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
                        
                        if st.button("Update Progress", key=f"update_{goal['id']}"):
                            st.session_state.selected_goal = goal["id"]
                            st.session_state.update_goal = True
                            st.experimental_rerun()
                        
                        if st.button("Get Insights", key=f"insights_{goal['id']}"):
                            st.session_state.selected_goal = goal["id"]
                            st.session_state.show_insights = True
                            st.experimental_rerun()
                        
                        if st.button("Mark Complete", key=f"complete_{goal['id']}"):
                            result = goal_manager.mark_goal_completed(goal["id"])
                            if result["success"]:
                                st.success("Goal marked as complete!")
                                st.experimental_rerun()
                            else:
                                st.error(f"Error: {result['message']}")
            
            # Completed goals section
            if completed_goals:
                st.markdown('<h3 class="section-header">Completed Goals</h3>', unsafe_allow_html=True)
                
                with st.expander("View Completed Goals"):
                    for goal in completed_goals:
                        st.markdown(f'<div class="card">', unsafe_allow_html=True)
                        st.markdown(f"### {goal['purpose']}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Target:** {UIHelper.format_currency(goal['target_amount'])}")
                            st.markdown(f"**Saved:** {UIHelper.format_currency(goal['current_amount'])}")
                        
                        with col_b:
                            completed_date = goal.get("completed_date", "Unknown")
                            if isinstance(completed_date, str) and completed_date != "Unknown":
                                completed_date = datetime.fromisoformat(completed_date.replace("Z", "+00:00")).strftime("%b %d, %Y")
                            
                            st.markdown(f"**Completed on:** {completed_date}")
                            
                            deadline = datetime.fromisoformat(goal["deadline"].replace("Z", "+00:00")).strftime("%b %d, %Y")
                            st.markdown(f"**Original Deadline:** {deadline}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle update goal if selected
        if hasattr(st.session_state, 'update_goal') and st.session_state.update_goal:
            goal_id = st.session_state.selected_goal
            goal = goal_manager.get_goal_by_id(goal_id)
            
            if goal:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"## Update Progress for {goal['purpose']}")
                
                current_amount = st.number_input(
                    "Current Amount Saved ($)",
                    min_value=0.0,
                    value=float(goal['current_amount']),
                    step=10.0
                )
                
                if st.button("Save Progress"):
                    result = goal_manager.update_goal_progress(goal_id, current_amount)
                    
                    if result["success"]:
                        st.success("Progress updated successfully!")
                        st.session_state.update_goal = False
                        st.experimental_rerun()
                    else:
                        st.error(f"Error: {result['message']}")
                
                if st.button("Cancel", key="cancel_update"):
                    st.session_state.update_goal = False
                    st.experimental_rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle goal insights if selected
        if hasattr(st.session_state, 'show_insights') and st.session_state.show_insights:
            goal_id = st.session_state.selected_goal
            
            with st.spinner("Generating goal insights..."):
                insights = ai_insight_generator.generate_goal_suggestions(goal_id)
                
                if insights["success"]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("## Goal Insights")
                    st.markdown(insights["suggestions"])
                    
                    if st.button("Close", key="close_insights"):
                        st.session_state.show_insights = False
                        st.experimental_rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Could not generate insights: {insights['message']}")
                    st.session_state.show_insights = False
    
    # Set New Goal Tab
    with tabs[1]:
        st.markdown('<h2 class="section-header">Create a New Financial Goal</h2>', unsafe_allow_html=True)
        
        # Natural language input
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Express Your Goal")
        st.markdown("Describe your financial goal in natural language, e.g., *\"I want to save $5,000 for a vacation by December\"*")
        
        goal_input = st.text_input("Describe your goal:", key="goal_input")
        
        if st.button("Create Goal", key="create_goal_nl"):
            if goal_input:
                # Process with NLP
                goal_details = nlp_processor.extract_goal_details(goal_input)
                
                if goal_details:
                    # Add goal to database
                    result = goal_manager.add_goal(
                        target_amount=goal_details["target_amount"],
                        purpose=goal_details["purpose"],
                        deadline=goal_details["deadline"]
                    )
                    
                    if result["success"]:
                        st.success(f"Created goal: Save {UIHelper.format_currency(goal_details['target_amount'])} for {goal_details['purpose']} by {goal_details['deadline']}")
                        st.experimental_rerun()
                    else:
                        st.error(f"Could not create goal: {result['message']}")
                else:
                    st.error("Could not understand goal details. Please try again with a clearer description.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Manual goal creation form
        st.markdown('<h3 class="section-header">Manual Goal Creation</h3>', unsafe_allow_html=True)
        
        with st.form("goal_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                target_amount = st.number_input("Target Amount ($)", min_value=1.0, step=100.0)
                deadline = st.date_input("Deadline", value=datetime.now() + timedelta(days=30))
            
            with col2:
                purpose = st.text_input("Purpose")
                
            submit_form = st.form_submit_button("Create Goal")
        
        if submit_form:
            result = goal_manager.add_goal(
                target_amount=target_amount,
                purpose=purpose,
                deadline=deadline.strftime("%Y-%m-%d")
            )
            
            if result["success"]:
                st.success(f"Created goal: Save {UIHelper.format_currency(target_amount)} for {purpose}")
                st.experimental_rerun()
            else:
                st.error(f"Could not create goal: {result['message']}")
    
    # Goal Insights Tab
    with tabs[2]:
        st.markdown('<h2 class="section-header">Goal Insights</h2>', unsafe_allow_html=True)
        
        # Get active goals
        active_goals = goal_manager.get_active_goals()
        
        if not active_goals:
            st.info("You don't have any active goals. Create a goal to get personalized insights!")
        else:
            # Goal selection
            goal_options = {f"{goal['purpose']} (${goal['target_amount']:,.2f})": goal["id"] for goal in active_goals}
            selected_goal = st.selectbox(
                "Select a goal to analyze",
                options=list(goal_options.keys())
            )
            
            if selected_goal:
                goal_id = goal_options[selected_goal]
                
                with st.spinner("Analyzing your goal and generating insights..."):
                    # Get goal progress
                    progress = goal_manager.get_goal_progress(goal_id)
                    
                    if progress["success"]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        
                        # Display progress metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            UIHelper.create_metric_card(
                                "Progress",
                                UIHelper.format_percentage(progress["progress_percentage"]),
                                help_text=f"{UIHelper.format_currency(progress['goal']['current_amount'])} of {UIHelper.format_currency(progress['goal']['target_amount'])}"
                            )
                        
                        with col2:
                            UIHelper.create_metric_card(
                                "Time Remaining",
                                f"{progress['days_remaining']} days",
                                help_text=f"Deadline: {datetime.fromisoformat(progress['goal']['deadline'].replace('Z', '+00:00')).strftime('%b %d, %Y')}"
                            )
                        
                        with col3:
                            UIHelper.create_metric_card(
                                "Required Monthly",
                                UIHelper.format_currency(progress["required_monthly"]),
                                help_text="To reach your goal on time"
                            )
                        
                        # Progress visualization
                        UIHelper.create_progress_bar(
                            "Goal Progress",
                            progress["goal"]["current_amount"],
                            progress["goal"]["target_amount"],
                            progress["progress_percentage"]
                        )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Get AI insights
                        insights = ai_insight_generator.generate_goal_suggestions(goal_id)
                        
                        if insights["success"]:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown("### AI Suggestions")
                            st.markdown(insights["suggestions"])
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error(f"Could not generate insights: {insights['message']}")
                    else:
                        st.error(f"Could not get goal progress: {progress['message']}")

# Settings Page
elif st.session_state.page == "settings":
    st.markdown('<h1 class="main-header">Settings</h1>', unsafe_allow_html=True)
    
    # Create tabs for different settings
    tabs = st.tabs(["Categories", "Notifications", "Data Management"])
    
    # Categories Tab
    with tabs[0]:
        st.markdown('<h2 class="section-header">Expense Categories</h2>', unsafe_allow_html=True)
        st.markdown("Customize your expense categories and their colors.")
        
        # Display current categories
        categories = [
            "Groceries", "Dining", "Entertainment", "Utilities", 
            "Transport", "Housing", "Healthcare", "Shopping", 
            "Education", "Travel", "Subscriptions", "Personal", "Other"
        ]
        
        for category in categories:
            color = ColorPalette.get_category_color(category)
            st.markdown(
                f'<div style="display:flex; align-items:center; margin-bottom:10px;">'
                f'<div style="width:20px; height:20px; background-color:{color}; border-radius:50%; margin-right:10px;"></div>'
                f'<span>{category}</span>'
                '</div>',
                unsafe_allow_html=True
            )
    
    # Notifications Tab
    with tabs[1]:
        st.markdown('<h2 class="section-header">Notifications</h2>', unsafe_allow_html=True)
        st.markdown("Configure your notification preferences.")
        
        # Notification settings
        st.checkbox("Email notifications", value=True)
        st.checkbox("Goal progress updates", value=True)
        st.checkbox("Monthly summary", value=True)
        st.checkbox("Budget alerts", value=True)
        
        # Notification frequency
        frequency = st.selectbox(
            "Update frequency",
            options=["Daily", "Weekly", "Monthly"]
        )
    
    # Data Management Tab
    with tabs[2]:
        st.markdown('<h2 class="section-header">Data Management</h2>', unsafe_allow_html=True)
        st.markdown("Manage your financial data.")
        
        # Export data
        st.markdown("### Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Expenses (CSV)"):
                # This would need additional logic to create CSV
                st.info("Export functionality will be added soon.")
        
        with col2:
            if st.button("Export Goals (CSV)"):
                # This would need additional logic to create CSV
                st.info("Export functionality will be added soon.")
        
        # Data deletion
        st.markdown("### Delete Data")
        st.warning("Warning: This action cannot be undone!")
        
        if st.button("Delete All Data", type="secondary"):
            # This would need additional confirmation and logic
            st.error("Please contact support to delete your data.")

# Add footer
st.markdown("""
<div style='position: fixed; bottom: 0; width: 100%; background-color: #f8f9fa; padding: 10px; text-align: center;'>
    <p style='margin: 0; font-size: 12px; color: #666;'>
        Personal Finance Assistant © 2024 | Built with ❤️ using Streamlit
    </p>
</div>
""", unsafe_allow_html=True)