import streamlit as st
import os
import re
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import markdown
import time
from datetime import datetime

# Import our backend modules
# In production, this would be properly packaged
# For now, we'll assume the backend code is imported as modules
from finance_education import (
    curriculum_agent, 
    assessment_agent, 
    parse_curriculum,
    get_topic_content,
    generate_topic_quiz,
    generate_custom_assessment
)

# Configuration
st.set_page_config(
    page_title="Personal Finance Education Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .financial-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .green-header {
        color: #2E7D32;
        font-weight: bold;
    }
    .progress-container {
        margin-bottom: 10px;
    }
    .quiz-container {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    .explanation-container {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        border-left: 3px solid #81C784;
    }
    .correct-answer {
        color: #2E7D32;
        font-weight: bold;
    }
    .incorrect-answer {
        color: #C62828;
        font-weight: bold;
    }
    .source-container {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 10px;
        font-size: 0.9em;
        margin-top: 20px;
    }
    .sidebar-menu {
        padding: 10px;
        border-radius: 5px;
        background-color: #e8f5e9;
        margin-bottom: 10px;
    }
    .sidebar-heading {
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 10px;
    }
    .finance-badge {
        background-color: #4CAF50;
        color: white;
        padding: 4px 8px;
        border-radius: 10px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_level' not in st.session_state:
    st.session_state.current_level = 1
if 'current_week' not in st.session_state:
    st.session_state.current_week = 1
if 'current_subtopic' not in st.session_state:
    st.session_state.current_subtopic = None
if 'quiz_results' not in st.session_state:
    st.session_state.quiz_results = {}
if 'completed_topics' not in st.session_state:
    st.session_state.completed_topics = set()
if 'user_notes' not in st.session_state:
    st.session_state.user_notes = {}
if 'quiz_state' not in st.session_state:
    st.session_state.quiz_state = None
if 'last_quiz_score' not in st.session_state:
    st.session_state.last_quiz_score = None

# Load curriculum data
# In a real app, this would come from your backend
def load_curriculum():
    """Load the curriculum data"""
    curriculum_md = """
# Personal Finance Curriculum Roadmap

## Level 1: Financial Foundations (Weeks 1-4)
- **Week 1: Money Basics**
  - Understanding income and expenses
  - Creating a personal balance sheet
  - Setting financial goals
  - Financial mindset fundamentals
  
- **Week 2: Budgeting Fundamentals**
  - Creating your first budget
  - Expense tracking methods
  - The 50/30/20 rule
  - Tools for budget management
  
- **Week 3: Banking and Cash Flow**
  - Types of bank accounts
  - Managing checking and savings accounts
  - Understanding fees and interest
  - Emergency funds fundamentals
  
- **Week 4: Debt Management Basics**
  - Understanding good vs. bad debt
  - Credit cards and interest rates
  - Student loans overview
  - Debt repayment strategies

## Level 2: Building Financial Security (Weeks 5-8)
- **Week 5: Credit Building**
  - Understanding credit scores
  - Credit reports and monitoring
  - Improving your credit score
  - Identity protection basics
  
- **Week 6: Insurance Fundamentals**
  - Types of insurance (health, auto, renters/homeowners)
  - How insurance works
  - Determining appropriate coverage levels
  - Insurance cost optimization
  
- **Week 7: Emergency Preparedness**
  - Building a 3-6 month emergency fund
  - Financial disaster planning
  - Insurance as protection
  - Unexpected expense management
  
- **Week 8: Saving Strategies**
  - High-yield savings accounts
  - Automating your savings
  - Saving for short and medium-term goals
  - Psychology of saving

## Level 3: Growing Your Wealth (Weeks 9-14)
- **Week 9: Investing Fundamentals**
  - Investment vehicles overview
  - Risk vs. return
  - Power of compound interest
  - Time horizon and investing goals
  
- **Week 10: Retirement Basics**
  - 401(k)s and workplace retirement plans
  - IRAs (Traditional & Roth)
  - The power of tax-advantaged accounts
  - Retirement planning concepts
"""
    # Transform the markdown into a structured format
    # This is a simplified version of the parse_curriculum function
    
    curriculum = []
    current_level = None
    current_week = None
    
    # Regular expressions to parse curriculum
    level_pattern = r"## Level (\d+): (.+?) \(Weeks (\d+)-(\d+)\)"
    week_pattern = r"\*\*Week (\d+): (.+?)\*\*"
    subtopic_pattern = r"  - (.+)"
    
    lines = curriculum_md.split('\n')
    
    for line in lines:
        level_match = re.search(level_pattern, line)
        if level_match:
            current_level = int(level_match.group(1))
            level_title = level_match.group(2)
            continue
            
        week_match = re.search(week_pattern, line)
        if week_match and current_level:
            current_week = int(week_match.group(1))
            week_title = week_match.group(2)
            curriculum.append({
                'level': current_level,
                'week': current_week,
                'title': week_title,
                'subtopics': []
            })
            continue
            
        subtopic_match = re.search(subtopic_pattern, line)
        if subtopic_match and current_level and current_week and curriculum:
            subtopic = subtopic_match.group(1)
            curriculum[-1]['subtopics'].append(subtopic)
            
    return curriculum

curriculum = load_curriculum()

# Mock functions for demonstration purposes
# In a real app, these would call your backend functions
def get_topic_content_mock(level, week, subtopic=None):
    """Mock function to get topic content"""
    # Simulate loading time
    with st.spinner("Retrieving educational content..."):
        time.sleep(2)
    
    topic = next((t for t in curriculum if t['level'] == level and t['week'] == week), None)
    if not topic:
        return "Topic not found in curriculum"
    
    if subtopic:
        content = f"""
## {subtopic}

Personal finance begins with understanding the flow of money in your life. Income represents money coming in, while expenses are money going out. The relationship between these two determines your financial health.

### Types of Income
- **Earned Income:** Money from jobs and work
- **Passive Income:** Money from investments
- **Portfolio Income:** Money from appreciation

### Common Expense Categories
- Housing (25-35% of income)
- Transportation (10-15%)
- Food (10-15%)
- Utilities (5-10%)
- Insurance (10-25%)
- Debt payments (15% or less is recommended)
- Savings (15-20% ideally)
- Entertainment/Discretionary (5-10%)

### Income vs. Expenses Analysis
Your "cash flow" is the difference between income and expenses. Positive cash flow means you're spending less than you earn, which is essential for financial health.

```
Cash Flow = Total Income - Total Expenses
```

#### Example:
Monthly Income: $4,000
Monthly Expenses: $3,500
Cash Flow: $500 positive

The goal is to maintain positive cash flow and gradually increase the gap between income and expenses.

### Sources:
1. Personal Finance for Dummies (Wiley Publishing)
2. Dave Ramsey's Complete Guide to Money
3. Your Money or Your Life by Vicki Robin
"""
    else:
        subtopics_list = "\n".join([f"- {s}" for s in topic['subtopics']])
        content = f"""
# {topic['title']}

This week focuses on understanding the fundamental aspects of money management. You'll learn the basic building blocks that form the foundation of personal finance.

## Topics covered this week:
{subtopics_list}

These topics will help you develop a clear understanding of your current financial situation and establish a framework for making better financial decisions.

### Key Learning Objectives:
1. Understand how money flows in and out of your life
2. Learn to create a simple personal balance sheet
3. Develop clear and achievable financial goals
4. Foster a healthy mindset around money and finances

### Sources:
1. Personal Finance for Dummies (Wiley Publishing)
2. Dave Ramsey's Complete Guide to Money
3. Your Money or Your Life by Vicki Robin
"""
    
    return content

def generate_topic_quiz_mock(level, week, subtopic=None):
    """Mock function to generate a quiz"""
    # Simulate loading time
    with st.spinner("Generating quiz questions..."):
        time.sleep(2)
    
    topic = next((t for t in curriculum if t['level'] == level and t['week'] == week), None)
    if not topic:
        return "Topic not found"
    
    if subtopic:
        topic_title = f"{subtopic} in {topic['title']}"
    else:
        topic_title = topic['title']
    
    # Mock quiz with 5 questions
    quiz = f"""
Question: Which of the following is NOT typically considered a type of income?
A. Earned income
B. Passive income
C. Expense income
D. Portfolio income
Correct Answer: C
Explanation: Expense income is not a recognized income type. The main income types are earned (from work), passive (from investments requiring little effort), and portfolio (from investments like stocks or bonds).

Question: What is the recommended percentage of income that should go toward housing expenses?
A. 10-15%
B. 25-35%
C. 40-50%
D. 50-60% 
Correct Answer: B
Explanation: Financial experts typically recommend allocating 25-35% of your income toward housing expenses, including mortgage/rent, utilities, and maintenance.

Question: How is cash flow calculated?
A. Total Assets - Total Liabilities
B. Total Income / Total Expenses
C. Total Income - Total Expenses
D. Total Expenses - Total Income
Correct Answer: C
Explanation: Cash flow is calculated by subtracting your total expenses from your total income, showing whether you have money left over (positive cash flow) or are spending more than you earn (negative cash flow).

Question: Which of the following would be considered a sign of good financial health?
A. Consistently increasing credit card balances
B. Spending exactly what you earn each month
C. Having positive cash flow each month
D. Having a diverse portfolio of debt
Correct Answer: C
Explanation: Having positive cash flow (spending less than you earn) is a clear sign of good financial health, as it allows you to save and invest for the future.

Question: What is an appropriate guideline for debt payments in your budget?
A. 5% or less of income
B. 15% or less of income
C. 30% or less of income
D. 50% or less of income
Correct Answer: B
Explanation: Financial experts typically recommend keeping total debt payments (excluding mortgage) to 15% or less of your monthly income to maintain financial health.
"""
    return quiz

def generate_custom_assessment_mock(topic):
    """Mock function to generate a custom assessment"""
    # Simulate loading time
    with st.spinner(f"Creating custom assessment on {topic}..."):
        time.sleep(3)
    
    assessment = f"""
# Custom Assessment: {topic}

## Multiple Choice Questions

Question 1: What is the primary benefit of using the debt snowball method?
A. It minimizes total interest paid
B. It provides psychological motivation by eliminating small debts first
C. It improves your credit score faster
D. It requires less total money to implement
Correct Answer: B
Explanation: The debt snowball method focuses on paying off smallest debts first regardless of interest rate, providing psychological wins that keep you motivated.

Question 2: Which debt repayment strategy is most efficient mathematically?
A. Debt snowball
B. Debt avalanche
C. Debt consolidation
D. Minimum payment method
Correct Answer: B
Explanation: The debt avalanche method (paying highest interest rate debts first) saves the most money in interest over time, making it mathematically optimal.

Question 3: In the debt avalanche method, which debts do you target first?
A. Smallest balance debts
B. Newest debts
C. Highest interest rate debts
D. Largest balance debts
Correct Answer: C
Explanation: The debt avalanche method prioritizes debts with the highest interest rates first, regardless of balance, to minimize interest payments.

Question 4: When might the debt snowball be more effective than the avalanche method?
A. When you have mostly high-interest debt
B. When you need psychological wins to stay motivated
C. When you have excellent financial discipline
D. When you have primarily large debts
Correct Answer: B
Explanation: The debt snowball can be more effective when motivation is a challenge, as quick wins from paying off small debts provide psychological momentum.

Question 5: What is a potential drawback of the debt avalanche method?
A. It costs more in total interest
B. It may take longer to feel progress if high-interest debts are large
C. It damages your credit score
D. It requires more complex calculations
Correct Answer: B
Explanation: A drawback of the avalanche method is that it might take longer to pay off your first debt if your highest-interest debts also have large balances, which can feel discouraging.

## Case Study: Maria's Debt Dilemma

Maria has the following debts:
- Credit Card A: $2,000 balance at 22% APR
- Credit Card B: $5,000 balance at 18% APR
- Personal Loan: $1,000 balance at 10% APR
- Student Loan: $15,000 balance at 6% APR

She has $500 extra each month to put toward debt repayment beyond minimum payments.

Analysis Question 1: If Maria uses the debt snowball method, in what order would she pay off her debts? Calculate how long it would take to pay off the first debt (assuming minimum payments of $50 on each debt).

Analysis Question 2: If Maria uses the debt avalanche method, in what order would she pay off her debts? Calculate how much interest she would save compared to the snowball method over the complete repayment period.

## Personal Reflection

Reflection Question: Consider your own debt situation or a hypothetical one. Which method—debt snowball or avalanche—would be more suitable for your personality and financial circumstances? Explain your reasoning and describe how you would implement your chosen strategy.
"""
    return assessment

# UI Components
def render_header():
    """Render the app header"""
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2534/2534679.png", width=80)
    with col2:
        st.title("Personal Finance Education Platform")
        st.markdown("<p style='font-size: 1.2em; color: #4CAF50;'>Your journey to financial literacy and freedom</p>", unsafe_allow_html=True)
    st.divider()

def render_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        st.markdown("<div class='sidebar-heading'>NAVIGATION</div>", unsafe_allow_html=True)
        
        # Dashboard button
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.current_subtopic = None
            st.session_state.quiz_state = None
        
        st.markdown("<div class='sidebar-heading'>CURRICULUM</div>", unsafe_allow_html=True)
        
        # Group by level
        levels = set(item['level'] for item in curriculum)
        for level in sorted(levels):
            level_items = [item for item in curriculum if item['level'] == level]
            level_title = f"Level {level}: {get_level_title(level)}"
            
            with st.expander(level_title, level == st.session_state.current_level):
                for item in level_items:
                    week_title = f"Week {item['week']}: {item['title']}"
                    
                    # Check if completed
                    is_completed = (level, item['week'], None) in st.session_state.completed_topics
                    week_icon = "✅ " if is_completed else "📘 "
                    
                    # Week button
                    if st.button(f"{week_icon}{week_title}", key=f"week_{level}_{item['week']}", use_container_width=True):
                        st.session_state.current_level = level
                        st.session_state.current_week = item['week']
                        st.session_state.current_subtopic = None
                        st.session_state.quiz_state = None
                        st.rerun()
        
        st.markdown("<div class='sidebar-heading'>TOOLS</div>", unsafe_allow_html=True)
        
        # Custom assessment button
        if st.button("🧩 Custom Assessment", use_container_width=True):
            st.session_state.current_level = None
            st.session_state.current_week = None
            st.session_state.current_subtopic = None
            st.session_state.quiz_state = "custom"
            st.rerun()
        
        # Notes button
        if st.button("📝 My Notes", use_container_width=True):
            st.session_state.current_level = None
            st.session_state.current_week = None
            st.session_state.current_subtopic = None
            st.session_state.quiz_state = "notes"
            st.rerun()
        
        # Progress tracking
        st.markdown("<div class='sidebar-heading'>YOUR PROGRESS</div>", unsafe_allow_html=True)
        
        # Calculate progress
        total_topics = sum(len(item['subtopics']) + 1 for item in curriculum)  # +1 for the main topic
        completed = len(st.session_state.completed_topics)
        progress_pct = (completed / total_topics) * 100 if total_topics > 0 else 0
        
        st.progress(progress_pct / 100)
        st.markdown(f"<div style='text-align: center;'>{progress_pct:.1f}% Complete</div>", unsafe_allow_html=True)
        
        # Quiz performance
        if st.session_state.quiz_results:
            avg_score = sum(st.session_state.quiz_results.values()) / len(st.session_state.quiz_results) * 100
            st.markdown(f"<div style='text-align: center; margin-top: 10px;'>Average Quiz Score: {avg_score:.1f}%</div>", unsafe_allow_html=True)
        
        # Latest badge earned
        if completed > 0:
            badge_level = "Beginner" if completed < 10 else "Intermediate" if completed < 20 else "Advanced"
            st.markdown(f"""
            <div style='text-align: center; margin-top: 20px;'>
                <div style='font-weight: bold; margin-bottom: 5px;'>Latest Badge:</div>
                <div class='finance-badge' style='display: inline-block; padding: 5px 10px;'>{badge_level} Finance Explorer</div>
            </div>
            """, unsafe_allow_html=True)

def get_level_title(level):
    """Get the title for a level"""
    level_titles = {
        1: "Financial Foundations",
        2: "Building Financial Security",
        3: "Growing Your Wealth",
        4: "Advanced Financial Planning",
        5: "Mastery and Specialization"
    }
    return level_titles.get(level, "Unknown Level")

def render_dashboard():
    """Render the dashboard view"""
    st.header("Your Financial Education Dashboard")
    
    # Progress overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='financial-card'>", unsafe_allow_html=True)
        st.subheader("Learning Progress")
        
        # Calculate level-by-level progress
        progress_data = []
        for level in sorted(set(item['level'] for item in curriculum)):
            level_items = [item for item in curriculum if item['level'] == level]
            total_level_topics = sum(len(item['subtopics']) + 1 for item in level_items)
            
            completed_level_topics = sum(
                1 for topic in st.session_state.completed_topics 
                if topic[0] == level
            )
            
            level_progress = (completed_level_topics / total_level_topics) * 100 if total_level_topics > 0 else 0
            progress_data.append({
                "Level": f"Level {level}",
                "Progress": level_progress
            })
        
        # Create progress chart
        progress_df = pd.DataFrame(progress_data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x="Level", y="Progress", data=progress_df, palette="Greens")
        plt.title("Progress by Level (%)")
        plt.ylabel("Completion (%)")
        plt.ylim(0, 100)
        
        # Display the chart
        st.pyplot(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='financial-card'>", unsafe_allow_html=True)
        st.subheader("Quiz Performance")
        
        if st.session_state.quiz_results:
            # Prepare quiz results data
            quiz_data = []
            for (level, week, subtopic), score in st.session_state.quiz_results.items():
                topic = next((t['title'] for t in curriculum if t['level'] == level and t['week'] == week), "Custom")
                label = f"{topic}" if not subtopic else f"{subtopic}"
                quiz_data.append({
                    "Topic": label[:15] + "..." if len(label) > 15 else label,
                    "Score": score * 100
                })
            
            # Create quiz performance chart
            quiz_df = pd.DataFrame(quiz_data[-5:])  # Show last 5 quizzes
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x="Topic", y="Score", data=quiz_df, palette="Greens")
            plt.title("Recent Quiz Scores (%)")
            plt.ylabel("Score (%)")
            plt.ylim(0, 100)
            plt.xticks(rotation=30, ha='right')
            
            # Display the chart
            st.pyplot(fig)
        else:
            st.info("No quiz results yet. Complete a quiz to see your performance.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Recommended next topics
    st.markdown("<div class='financial-card'>", unsafe_allow_html=True)
    st.subheader("Recommended Next Steps")
    
    # Find incomplete topics
    incomplete_topics = []
    for item in curriculum:
        level, week = item['level'], item['week']
        
        # Check if main topic is incomplete
        if (level, week, None) not in st.session_state.completed_topics:
            incomplete_topics.append((level, week, None, item['title']))
        
        # Check subtopics
        for subtopic in item['subtopics']:
            if (level, week, subtopic) not in st.session_state.completed_topics:
                incomplete_topics.append((level, week, subtopic, item['title']))
    
    # Filter out None values before sorting
    incomplete_topics = [topic for topic in incomplete_topics if topic is not None]
    
    # Sort by level and week (assuming level and week are the first two elements of the tuple)
    incomplete_topics.sort(key=lambda x: (x[0], x[1]))  # Sort by level and week
    
    # Display recommendations (first 3)
    if incomplete_topics:
        for i, (level, week, subtopic, title) in enumerate(incomplete_topics[:3]):
            with st.container():
                col1, col2, col3 = st.columns([1, 4, 1])
                with col1:
                    st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>{i+1}</h3>", unsafe_allow_html=True)
                with col2:
                    if subtopic:
                        st.markdown(f"**{subtopic}** in Week {week}: {title}")
                    else:
                        st.markdown(f"**Week {week}: {title}** overview")
                with col3:
                    if st.button("Start →", key=f"rec_{level}_{week}_{subtopic}", type="primary"):
                        st.session_state.current_level = level
                        st.session_state.current_week = week
                        st.session_state.current_subtopic = subtopic
                        st.session_state.quiz_state = None
                        st.rerun()
    else:
        st.success("Congratulations! You've completed all topics in the curriculum.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Recent activity
    st.markdown("<div class='financial-card'>", unsafe_allow_html=True)
    st.subheader("Financial Tip of the Day")
    
    # Rotate tips based on day of year
    tips = [
        "Pay yourself first - automatically save a portion of each paycheck.",
        "Track your expenses for a month to identify spending leaks.",
        "Aim to keep housing costs under 30% of your income.",
        "Build an emergency fund with 3-6 months of essential expenses.",
        "When investing, focus on asset allocation more than individual picks.",
        "Compound interest works - starting early matters more than the amount.",
        "Use the 50/30/20 rule: 50% needs, 30% wants, 20% savings/debt repayment.",
        "Credit cards should be convenience tools, not loans.",
        "Review your insurance coverage annually as your life changes.",
        "Retirement accounts like 401(k)s and IRAs provide valuable tax advantages."
    ]
    
    tip_index = datetime.now().timetuple().tm_yday % len(tips)
    st.markdown(f"""
    <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
        <div style='font-weight: bold; color: #2E7D32;'>💡 Today's Tip:</div>
        <div style='margin-top: 5px;'>{tips[tip_index]}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_topic_content():
    """Render the content for the current topic"""
    # Get current topic info
    current_topic = next((t for t in curriculum 
                        if t['level'] == st.session_state.current_level 
                        and t['week'] == st.session_state.current_week), None)
    
    if not current_topic:
        st.error("Topic not found")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "📝 Notes", "📊 Assessment"])
    
    with tab1:
        # Topic header
        level_title = get_level_title(st.session_state.current_level)
        st.header(f"Level {st.session_state.current_level}: {level_title}")
        st.subheader(f"Week {st.session_state.current_week}: {current_topic['title']}")
        
        # Mark complete button
        col1, col2 = st.columns([6, 1])
        with col2:
            complete_key = (st.session_state.current_level, st.session_state.current_week, st.session_state.current_subtopic)
            is_complete = complete_key in st.session_state.completed_topics
            
            if is_complete:
                st.success("Completed!")
            else:
                if st.button("Mark Complete", type="primary"):
                    st.session_state.completed_topics.add(complete_key)
                    st.success("Marked as complete!")
                    st.rerun()
        
        # Subtopic navigation if no specific subtopic is selected
        if st.session_state.current_subtopic is None:
            st.markdown("<div class='financial-card'>", unsafe_allow_html=True)
            
            # Display main topic content
            content = get_topic_content_mock(st.session_state.current_level, st.session_state.current_week)
            st.markdown(content)
            
            # Subtopics navigation
            st.markdown("<h3 class='green-header'>Explore Subtopics</h3>", unsafe_allow_html=True)
            
            # Create card-like buttons for each subtopic
            cols = st.columns(2)
            for i, subtopic in enumerate(current_topic['subtopics']):
                col = cols[i % 2]
                with col:
                    # Check if subtopic is completed
                    subtopic_key = (st.session_state.current_level, st.session_state.current_week, subtopic)
                    is_subtopic_complete = subtopic_key in st.session_state.completed_topics
                    complete_badge = "✅ " if is_subtopic_complete else ""
                    
                    st.markdown(f"""
                    <div style='background-color: white; padding: 15px; border-radius: 10px; 
                         margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); cursor: pointer;'
                         onclick="document.getElementById('subtopic_button_{i}').click();">
                        <div style='font-weight: bold; color: #2E7D32;'>{complete_badge}{subtopic}</div>
                        <div style='font-size: 0.8em; color: #666; margin-top: 5px;'>Click to explore</div>
                    </div>
                    <div style='display: none;'>
                        <button id='subtopic_button_{i}'></button>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Hidden button that gets triggered by the onclick
                    if st.button(f"Load {subtopic}", key=f"subtopic_button_{i}"):
                        st.session_state.current_subtopic = subtopic
                        st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Display specific subtopic content
            st.markdown("<div class='financial-card'>", unsafe_allow_html=True)
            
            # Back button
            if st.button("← Back to main topic"):
                st.session_state.current_subtopic = None
                st.rerun()
            
            # Display subtopic content
            content = get_topic_content_mock(
                st.session_state.current_level, 
                st.session_state.current_week,
                st.session_state.current_subtopic
            )
            st.markdown(content)
            
            # Related resources
            st.markdown("<h3 class='green-header'>Related Resources</h3>", unsafe_allow_html=True)
            st.markdown("""
            <ul>
                <li><a href="#" target="_blank">The Psychology of Money (Book Summary)</a></li>
                <li><a href="#" target="_blank">Interactive Budget Calculator Tool</a></li>
                <li><a href="#" target="_blank">Video: Understanding Income Types</a></li>
            </ul>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.header("Your Notes")
        
        # Construct note key
        note_key = (
            st.session_state.current_level,
            st.session_state.current_week,
            st.session_state.current_subtopic
        )
        
        # Get existing note or empty string
        current_note = st.session_state.user_notes.get(note_key, "")
        
        # Note editor
        new_note = st.text_area(
            "Add or edit your notes for this topic:",
            value=current_note,
            height=300,
            key=f"note_{note_key}"
        )
        
        # Save button
        if st.button("Save Notes", key=f"save_note_{note_key}"):
            st.session_state.user_notes[note_key] = new_note
            st.success("Notes saved successfully!")
    
    with tab3:
        st.header("Assessment")
        
        # Quiz state
        if st.session_state.quiz_state is None:
            # Quiz selection
            st.markdown("<div class='financial-card'>", unsafe_allow_html=True)
            st.subheader("Test your knowledge")
            
            quiz_type = st.radio(
                "Choose quiz type:",
                ["Topic Overview Quiz", "Subtopic Specific Quiz"],
                horizontal=True
            )
            
            if quiz_type == "Subtopic Specific Quiz" and st.session_state.current_subtopic is None:
                # Dropdown for subtopic selection
                selected_subtopic = st.selectbox(
                    "Select a subtopic to quiz on:",
                    current_topic['subtopics']
                )
            else:
                selected_subtopic = st.session_state.current_subtopic if quiz_type == "Subtopic Specific Quiz" else None
            
            if st.button("Start Quiz", type="primary"):
                st.session_state.quiz_state = "active"
                st.session_state.quiz_subtopic = selected_subtopic
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif st.session_state.quiz_state == "active":
            # Display active quiz
            st.markdown("<div class='quiz-container'>", unsafe_allow_html=True)
            
            # Get quiz content
            quiz_content = generate_topic_quiz_mock(
                st.session_state.current_level,
                st.session_state.current_week,
                st.session_state.quiz_subtopic
            )
            
            # Parse quiz questions
            questions = []
            current_question = {}
            
            for line in quiz_content.strip().split('\n'):
                if line.startswith("Question:"):
                    if current_question and 'question' in current_question:
                        questions.append(current_question)
                    current_question = {'question': line.replace("Question:", "").strip()}
                elif line.startswith("A."):
                    current_question['A'] = line.replace("A.", "").strip()
                elif line.startswith("B."):
                    current_question['B'] = line.replace("B.", "").strip()
                elif line.startswith("C."):
                    current_question['C'] = line.replace("C.", "").strip()
                elif line.startswith("D."):
                    current_question['D'] = line.replace("D.", "").strip()
                elif line.startswith("Correct Answer:"):
                    current_question['correct'] = line.replace("Correct Answer:", "").strip()
                elif line.startswith("Explanation:"):
                    current_question['explanation'] = line.replace("Explanation:", "").strip()
            
            # Add the last question
            if current_question and 'question' in current_question:
                questions.append(current_question)
            
            # Store user answers
            if 'user_answers' not in st.session_state:
                st.session_state.user_answers = {}
            
            # Display questions
            for i, q in enumerate(questions):
                st.subheader(f"Question {i+1}")
                st.write(q['question'])
                
                # Radio buttons for answers
                options = {
                    'A': q['A'],
                    'B': q['B'],
                    'C': q['C'],
                    'D': q['D']
                }
                
                answer = st.radio(
                    "Select your answer:",
                    options.keys(),
                    format_func=lambda x: f"{x}. {options[x]}",
                    key=f"q{i}"
                )
                
                # Store the answer
                st.session_state.user_answers[i] = answer
                
                st.markdown("<hr>", unsafe_allow_html=True)
            
            # Submit button
            if st.button("Submit Quiz", type="primary"):
                # Calculate score
                correct_count = sum(1 for i, q in enumerate(questions) 
                                  if st.session_state.user_answers.get(i) == q['correct'])
                score = correct_count / len(questions) if questions else 0
                
                # Store result
                result_key = (
                    st.session_state.current_level,
                    st.session_state.current_week,
                    st.session_state.quiz_subtopic
                )
                st.session_state.quiz_results[result_key] = score
                st.session_state.last_quiz_score = score
                
                # Change state to results
                st.session_state.quiz_state = "results"
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif st.session_state.quiz_state == "results":
            # Display quiz results
            st.markdown("<div class='quiz-container'>", unsafe_allow_html=True)
            
            # Get quiz content again
            quiz_content = generate_topic_quiz_mock(
                st.session_state.current_level,
                st.session_state.current_week,
                st.session_state.quiz_subtopic
            )
            
            # Parse quiz questions
            questions = []
            current_question = {}
            
            for line in quiz_content.strip().split('\n'):
                if line.startswith("Question:"):
                    if current_question and 'question' in current_question:
                        questions.append(current_question)
                    current_question = {'question': line.replace("Question:", "").strip()}
                elif line.startswith("A."):
                    current_question['A'] = line.replace("A.", "").strip()
                elif line.startswith("B."):
                    current_question['B'] = line.replace("B.", "").strip()
                elif line.startswith("C."):
                    current_question['C'] = line.replace("C.", "").strip()
                elif line.startswith("D."):
                    current_question['D'] = line.replace("D.", "").strip()
                elif line.startswith("Correct Answer:"):
                    current_question['correct'] = line.replace("Correct Answer:", "").strip()
                elif line.startswith("Explanation:"):
                    current_question['explanation'] = line.replace("Explanation:", "").strip()
            
            # Add the last question
            if current_question and 'question' in current_question:
                questions.append(current_question)
            
            # Show score
            score_percentage = st.session_state.last_quiz_score * 100
            if score_percentage >= 80:
                st.success(f"🎉 Great job! Your score: {score_percentage:.1f}%")
            elif score_percentage >= 60:
                st.warning(f"👍 Not bad! Your score: {score_percentage:.1f}%")
            else:
                st.error(f"📚 Keep learning! Your score: {score_percentage:.1f}%")
            
            # Review questions
            st.subheader("Review Questions")
            for i, q in enumerate(questions):
                user_answer = st.session_state.user_answers.get(i, None)
                is_correct = user_answer == q['correct']
                
                # Format for display
                result_icon = "✅" if is_correct else "❌"
                
                st.markdown(f"""
                <div style='background-color: {"#e8f5e9" if is_correct else "#ffebee"}; padding: 15px; 
                     border-radius: 10px; margin-bottom: 15px;'>
                    <div style='font-weight: bold;'>{result_icon} Question {i+1}</div>
                    <div style='margin: 10px 0;'>{q['question']}</div>
                    <div>
                        <span style='font-weight: bold;'>Your answer:</span> 
                        <span class='{"correct-answer" if is_correct else "incorrect-answer"}'>
                            {user_answer}. {q[user_answer] if user_answer else "Not answered"}
                        </span>
                    </div>
                    <div>
                        <span style='font-weight: bold;'>Correct answer:</span> 
                        <span class='correct-answer'>{q['correct']}. {q[q['correct']]}</span>
                    </div>
                    <div class='explanation-container'>{q['explanation']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Options for next steps
            st.subheader("Next Steps")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Retake Quiz", key="retake_quiz", type="secondary"):
                    st.session_state.quiz_state = "active"
                    if 'user_answers' in st.session_state:
                        del st.session_state.user_answers
                    st.rerun()
            
            with col2:
                if st.button("Back to Topic", key="back_to_topic", type="secondary"):
                    st.session_state.quiz_state = None
                    if 'user_answers' in st.session_state:
                        del st.session_state.user_answers
                    st.rerun()
            
            with col3:
                # Get next topic
                next_topic = None
                for item in curriculum:
                    if (item['level'] == st.session_state.current_level and 
                        item['week'] > st.session_state.current_week):
                        next_topic = (item['level'], item['week'])
                        break
                
                if not next_topic and st.session_state.current_level < max(item['level'] for item in curriculum):
                    # Move to next level, first week
                    next_level = st.session_state.current_level + 1
                    next_level_items = [item for item in curriculum if item['level'] == next_level]
                    if next_level_items:
                        next_topic = (next_level, min(item['week'] for item in next_level_items))
                
                if next_topic:
                    if st.button("Next Topic", key="next_topic", type="primary"):
                        st.session_state.current_level = next_topic[0]
                        st.session_state.current_week = next_topic[1]
                        st.session_state.current_subtopic = None
                        st.session_state.quiz_state = None
                        if 'user_answers' in st.session_state:
                            del st.session_state.user_answers
                        st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif st.session_state.quiz_state == "custom":
            # Custom assessment interface
            st.markdown("<div class='financial-card'>", unsafe_allow_html=True)
            st.subheader("Custom Financial Topic Assessment")
            
            st.write("Generate an assessment on any financial topic of your choice.")
            
            custom_topic = st.text_input(
                "Enter the financial topic you want to be assessed on:",
                placeholder="e.g., Debt snowball vs. avalanche method"
            )
            
            if st.button("Generate Assessment", type="primary") and custom_topic:
                with st.spinner(f"Creating assessment on {custom_topic}..."):
                    # Get custom assessment
                    assessment = generate_custom_assessment_mock(custom_topic)
                    
                    # Display assessment
                    st.markdown("<div class='quiz-container'>", unsafe_allow_html=True)
                    st.markdown(assessment)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Export option
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download as PDF",
                            data=assessment.encode(),
                            file_name=f"{custom_topic.replace(' ', '_')}_assessment.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        if st.button("Create Another Assessment"):
                            st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif st.session_state.quiz_state == "notes":
            # Notes management interface
            st.markdown("<div class='financial-card'>", unsafe_allow_html=True)
            st.subheader("Your Learning Notes")
            
            if st.session_state.user_notes:
                # Create tabs for organizing notes
                notes_tabs = st.tabs(["All Notes", "By Level", "Search"])
                
                with notes_tabs[0]:
                    # Display all notes
                    for (level, week, subtopic), note in st.session_state.user_notes.items():
                        if note.strip():  # Only show non-empty notes
                            topic = next((t['title'] for t in curriculum 
                                          if t['level'] == level and t['week'] == week), "Custom Topic")
                            
                            with st.expander(f"Level {level}, Week {week}: {topic}" + 
                                            (f" - {subtopic}" if subtopic else "")):
                                st.markdown(note)
                                
                                # Edit button
                                if st.button("Edit", key=f"edit_{level}_{week}_{subtopic}"):
                                    st.session_state.current_level = level
                                    st.session_state.current_week = week
                                    st.session_state.current_subtopic = subtopic
                                    st.session_state.quiz_state = None
                                    st.rerun()
                
                with notes_tabs[1]:
                    # Organize by level
                    for level in sorted(set(item['level'] for item in curriculum)):
                        level_notes = {k: v for k, v in st.session_state.user_notes.items() if k[0] == level}
                        
                        if level_notes:
                            st.markdown(f"### Level {level}: {get_level_title(level)}")
                            
                            for (l, week, subtopic), note in level_notes.items():
                                if note.strip():  # Only show non-empty notes
                                    topic = next((t['title'] for t in curriculum 
                                                if t['level'] == l and t['week'] == week), "Custom Topic")
                                    
                                    with st.expander(f"Week {week}: {topic}" + 
                                                    (f" - {subtopic}" if subtopic else "")):
                                        st.markdown(note)
                                        
                                        # Edit button
                                        if st.button("Edit", key=f"edit_level_{l}_{week}_{subtopic}"):
                                            st.session_state.current_level = l
                                            st.session_state.current_week = week
                                            st.session_state.current_subtopic = subtopic
                                            st.session_state.quiz_state = None
                                            st.rerun()
                
                with notes_tabs[2]:
                    # Search notes
                    search_term = st.text_input("Search your notes:")
                    
                    if search_term:
                        found_notes = False
                        for (level, week, subtopic), note in st.session_state.user_notes.items():
                            if search_term.lower() in note.lower():
                                found_notes = True
                                topic = next((t['title'] for t in curriculum 
                                            if t['level'] == level and t['week'] == week), "Custom Topic")
                                
                                with st.expander(f"Level {level}, Week {week}: {topic}" + 
                                                (f" - {subtopic}" if subtopic else "")):
                                    # Highlight search term
                                    highlighted_note = note.replace(
                                        search_term, 
                                        f"<span style='background-color: yellow;'>{search_term}</span>"
                                    )
                                    st.markdown(highlighted_note, unsafe_allow_html=True)
                                    
                                    # Edit button
                                    if st.button("Edit", key=f"edit_search_{level}_{week}_{subtopic}"):
                                        st.session_state.current_level = level
                                        st.session_state.current_week = week
                                        st.session_state.current_subtopic = subtopic
                                        st.session_state.quiz_state = None
                                        st.rerun()
                        
                        if not found_notes:
                            st.info(f"No notes found containing '{search_term}'")
                    else:
                        st.info("Enter a search term to find specific notes")
                
                # Export all notes
                st.download_button(
                    label="Export All Notes",
                    data=json.dumps(st.session_state.user_notes, indent=4),
                    file_name="finance_education_notes.json",
                    mime="application/json"
                )
            else:
                st.info("You haven't created any notes yet. Navigate to a topic and add notes from the Notes tab.")
            
            st.markdown("</div>", unsafe_allow_html=True)

def main():
    """Main application"""
    # Render header
    render_header()
    
    # Create layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Render sidebar
        render_sidebar()
    
    with col2:
        # Render content based on current state
        if st.session_state.current_level is None:
            # Render dashboard
            render_dashboard()
        else:
            # Render topic content
            render_topic_content()

if __name__ == "__main__":
    main()