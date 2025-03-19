"""
PERSONAL FINANCE TEACHING ASSISTANT - STREAMLIT FRONTEND

This module implements a Streamlit-based user interface for the
Personal Finance Teaching Assistant, connecting to the FastAPI backend.
"""

import os
import json
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "test-user-123")

# Page setup
st.set_page_config(
    page_title="Personal Finance Teaching Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def get_api_url(endpoint):
    """Get the full API URL for a specific endpoint."""
    return f"{API_URL}{endpoint}"

def make_api_request(endpoint, method="GET", data=None):
    """Make an API request to the backend service."""
    url = get_api_url(endpoint)
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            st.error(f"Unsupported request method: {method}")
            return None
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

def get_curriculum():
    """Get the full curriculum data from the API."""
    return make_api_request("/curriculum")

def get_user_progress(user_id):
    """Get the user's learning progress."""
    return make_api_request(f"/user/progress/{user_id}")

def ask_question(user_id, query, session_id=None):
    """Ask a question to the teaching assistant."""
    data = {
        "user_id": user_id,
        "query": query,
        "session_id": session_id
    }
    return make_api_request("/chat", method="POST", data=data)

def generate_quiz(user_id, topic, level=None):
    """Generate a quiz on a specific topic."""
    data = {
        "user_id": user_id,
        "topic": topic,
        "level": level
    }
    return make_api_request("/assessment/quiz", method="POST", data=data)

def submit_quiz_answers(user_id, quiz_id, answers):
    """Submit answers to a quiz."""
    data = {
        "user_id": user_id,
        "quiz_id": quiz_id,
        "answers": answers
    }
    return make_api_request("/assessment/submit-quiz", method="POST", data=data)

def get_learning_path(user_id, goal=None):
    """Get a personalized learning path."""
    data = {
        "user_id": user_id,
        "goal": goal
    }
    return make_api_request("/learning/path", method="POST", data=data)

def complete_module(user_id, module_id):
    """Mark a module as completed."""
    data = {
        "user_id": user_id,
        "module_id": module_id
    }
    return make_api_request("/learning/complete-module", method="POST", data=data)

# Initialize session state
def init_session_state():
    """Initialize session state variables if they don't already exist."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = DEFAULT_USER_ID
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    if "current_quiz" not in st.session_state:
        st.session_state.current_quiz = None
    
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    
    if "current_module" not in st.session_state:
        st.session_state.current_module = None

# Callback functions
def on_message_submit():
    """Process user message submission."""
    if st.session_state.user_message:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": st.session_state.user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send query to API
        response = ask_question(
            st.session_state.user_id, 
            st.session_state.user_message, 
            st.session_state.session_id
        )
        
        if response:
            # Save session ID
            st.session_state.session_id = response.get("session_id")
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.get("response", "Sorry, I couldn't process your request."),
                "timestamp": datetime.now().isoformat()
            })
        
        # Clear input
        st.session_state.user_message = ""

def on_quiz_submit():
    """Process quiz submission."""
    if st.session_state.current_quiz:
        quiz_id = st.session_state.current_quiz.get("quiz_id")
        
        # Submit answers to API
        results = submit_quiz_answers(
            st.session_state.user_id,
            quiz_id,
            st.session_state.quiz_answers
        )
        
        if results:
            # Display results
            st.session_state.quiz_results = results
            st.session_state.show_quiz_results = True
        
        # Reset current quiz
        st.session_state.current_quiz = None
        st.session_state.quiz_answers = {}

def on_module_complete(module_id):
    """Mark a module as completed."""
    result = complete_module(st.session_state.user_id, module_id)
    
    if result:
        st.session_state.module_completed = True
        st.session_state.next_module = result.get("next_module")

# UI Components
def render_sidebar():
    """Render the sidebar with user settings and navigation."""
    with st.sidebar:
        st.title("Finance Education")
        
        # User settings
        st.subheader("User Settings")
        user_id = st.text_input("User ID", value=st.session_state.user_id)
        if user_id != st.session_state.user_id:
            st.session_state.user_id = user_id
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Go to",
            options=["Chat", "Curriculum", "Learning Path", "Progress", "Quiz"],
            index=0
        )
        
        # Apply the page selection
        st.session_state.current_page = page
        
        # Custom financial goal input for learning path
        if page == "Learning Path":
            st.subheader("Learning Goal")
            st.session_state.goal = st.text_area(
                "Enter your financial goal",
                placeholder="e.g., I want to save for retirement, or I want to buy a house in 5 years"
            )
        
        # Quiz topic input
        if page == "Quiz":
            st.subheader("Quiz Topic")
            st.session_state.quiz_topic = st.text_input(
                "Enter quiz topic",
                placeholder="e.g., Budgeting, Investing, Retirement"
            )
            st.session_state.quiz_level = st.slider(
                "Select difficulty level",
                min_value=1,
                max_value=5,
                value=1,
                help="1 = Beginner, 5 = Expert"
            )
        
        # About section
        st.markdown("---")
        st.markdown(
            "**Personal Finance Teaching Assistant**\n\n"
            "Learn finance from basic to advanced levels with a personalized AI tutor."
        )

def render_chat_ui():
    """Render the chat interface."""
    st.header("💬 Chat with Your Finance Teacher")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Teacher:** {message['content']}")
    
    # Input for new message
    st.text_area(
        "Ask a question about personal finance",
        key="user_message",
        height=100,
        placeholder="e.g., What is compound interest? or How do I create a budget?",
        on_change=on_message_submit
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button("Ask", on_click=on_message_submit)

def render_curriculum_ui():
    """Render the curriculum view."""
    st.header("📚 Finance Education Curriculum")
    
    # Fetch curriculum data
    curriculum_data = get_curriculum()
    
    if not curriculum_data:
        st.error("Failed to load curriculum data.")
        return
    
    # Extract modules
    modules = curriculum_data.get("modules", [])
    
    # Group modules by level
    modules_by_level = {}
    for module in modules:
        level = module.get("level")
        if level not in modules_by_level:
            modules_by_level[level] = []
        modules_by_level[level].append(module)
    
    # Create tabs for each level
    if modules_by_level:
        tabs = st.tabs([f"Level {level}: {modules[0].get('level_title')}" 
                       for level, modules in sorted(modules_by_level.items())])
        
        # Populate each tab
        for i, (level, modules) in enumerate(sorted(modules_by_level.items())):
            with tabs[i]:
                st.subheader(f"{modules[0].get('level_title')}")
                st.markdown(f"*{next((m.get('level_description', '') for m in modules if 'level_description' in m), '')}*")
                
                # Display modules in columns
                cols = st.columns(len(modules))
                for j, module in enumerate(modules):
                    with cols[j]:
                        st.markdown(f"### {module.get('title')}")
                        st.markdown(f"*{module.get('description')}*")
                        
                        # Display topics
                        st.markdown("**Topics:**")
                        for topic in module.get("topics", []):
                            st.markdown(f"- {topic}")
                        
                        # Action buttons
                        st.button(
                            "Start Learning", 
                            key=f"start_{module.get('module_id')}",
                            on_click=lambda m=module: set_current_module(m)
                        )
    else:
        st.info("No curriculum modules found.")

def set_current_module(module):
    """Set the current module and navigate to it."""
    st.session_state.current_module = module
    st.session_state.current_page = "Module Details"

def render_module_details_ui():
    """Render the details of a specific module."""
    module = st.session_state.current_module
    
    if not module:
        st.error("No module selected.")
        return
    
    st.header(f"📖 {module.get('title')}")
    st.subheader(f"Level {module.get('level')}: {module.get('level_title')}")
    
    st.markdown(f"*{module.get('description')}*")
    
    # Display estimated time
    st.info(f"Estimated time: {module.get('estimated_time', 'Not specified')}")
    
    # Topics section
    st.subheader("Topics Covered")
    for topic in module.get("topics", []):
        st.markdown(f"- {topic}")
    
    # Exercises section
    st.subheader("Practical Exercises")
    for exercise in module.get("exercises", []):
        st.markdown(f"- {exercise}")
    
    # Actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Mark as Completed"):
            on_module_complete(module.get("module_id"))
    
    with col2:
        if st.button("Take a Quiz on This Topic"):
            st.session_state.quiz_topic = module.get("title")
            st.session_state.current_page = "Quiz"
    
    with col3:
        if st.button("Back to Curriculum"):
            st.session_state.current_module = None
            st.session_state.current_page = "Curriculum"
    
    # Display completion confirmation
    if hasattr(st.session_state, "module_completed") and st.session_state.module_completed:
        st.success("Module marked as completed!")
        
        if hasattr(st.session_state, "next_module") and st.session_state.next_module:
            next_module = st.session_state.next_module
            st.markdown(f"**Next recommended module:** {next_module.get('title')}")
            
            if st.button("Go to Next Module"):
                set_current_module(next_module)
                st.session_state.module_completed = False
                st.session_state.next_module = None

def render_learning_path_ui():
    """Render the personalized learning path view."""
    st.header("🗺️ Your Personal Learning Path")
    
    # Check if we have a goal
    goal = getattr(st.session_state, "goal", None)
    
    # Fetch learning path
    learning_path_data = get_learning_path(st.session_state.user_id, goal)
    
    if not learning_path_data:
        st.error("Failed to load learning path data.")
        return
    
    # Extract learning path
    learning_path = learning_path_data.get("learning_path", [])
    
    if not learning_path:
        st.info("No learning path generated yet.")
        return
    
    # Display the goal if provided
    if goal:
        st.subheader(f"Learning Path for Goal: '{goal}'")
    else:
        st.subheader("Recommended Learning Path")
    
    # Display learning path as a timeline
    for i, module in enumerate(learning_path):
        with st.expander(f"Step {i+1}: {module.get('title')}", expanded=i==0):
            st.markdown(f"*{module.get('description')}*")
            
            # Display reason for recommendation
            st.info(module.get('reason', 'Recommended module in your learning path.'))
            
            # Action button
            if st.button("View Module Details", key=f"view_{module.get('module_id')}"):
                # We need to fetch the full module data
                # For now, we'll use a simplified version
                st.session_state.current_module = module
                st.session_state.current_page = "Module Details"

def render_progress_ui():
    """Render the user's learning progress."""
    st.header("📊 Your Learning Progress")
    
    # Fetch user progress
    progress_data = get_user_progress(st.session_state.user_id)
    
    if not progress_data:
        st.error("Failed to load user progress data.")
        return
    
    # Extract progress information
    progress = progress_data.get("progress", {})
    
    # User level
    knowledge_level = progress.get("knowledge_level", 1)
    st.subheader(f"Current Knowledge Level: {knowledge_level}/5")
    
    # Progress bar
    st.progress(knowledge_level / 5, f"Level {knowledge_level}")
    
    # Completed modules
    completed_modules = progress.get("completed_modules", [])
    st.subheader("Completed Modules")
    
    if completed_modules:
        # Fetch curriculum to get module titles
        curriculum_data = get_curriculum()
        if curriculum_data:
            modules = {m["module_id"]: m for m in curriculum_data.get("modules", [])}
            
            # Create a table of completed modules
            data = []
            for module_id in completed_modules:
                module = modules.get(module_id, {"title": module_id, "level": "?"})
                data.append({
                    "Module ID": module_id,
                    "Title": module.get("title", module_id),
                    "Level": module.get("level", "?")
                })
            
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
    else:
        st.info("No modules completed yet.")
    
    # Quiz scores
    quiz_scores = progress.get("quiz_scores", {})
    st.subheader("Quiz Results")
    
    if quiz_scores:
        # Create a visualization of quiz scores
        data = [{"Quiz": quiz_id, "Score": score} for quiz_id, score in quiz_scores.items()]
        df = pd.DataFrame(data)
        
        # Bar chart of quiz scores
        fig = px.bar(
            df, 
            x="Quiz", 
            y="Score", 
            title="Your Quiz Scores",
            labels={"Quiz": "Quiz ID", "Score": "Score (%)"},
            range_y=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No quizzes taken yet.")

def render_quiz_ui():
    """Render the quiz interface."""
    st.header("🧠 Test Your Knowledge")
    
    # Check if we're showing quiz results
    if hasattr(st.session_state, "show_quiz_results") and st.session_state.show_quiz_results:
        render_quiz_results()
        return
    
    # Check if we already have a quiz loaded
    if st.session_state.current_quiz:
        render_quiz_questions()
        return
    
    # Quiz topic input
    topic = getattr(st.session_state, "quiz_topic", "")
    level = getattr(st.session_state, "quiz_level", 1)
    
    if not topic:
        st.info("Enter a topic in the sidebar to generate a quiz.")
        return
    
    # Generate quiz button
    if st.button("Generate Quiz"):
        quiz_data = generate_quiz(st.session_state.user_id, topic, level)
        
        if quiz_data:
            st.session_state.current_quiz = quiz_data
            st.session_state.quiz_answers = {}
            st.experimental_rerun()

def render_quiz_questions():
    """Render the questions for the current quiz."""
    quiz = st.session_state.current_quiz
    
    if not quiz:
        st.error("No quiz data available.")
        return
    
    st.subheader(f"Quiz: {quiz.get('topic')}")
    st.markdown(f"*Difficulty Level: {quiz.get('level')}/5*")
    
    # Display questions
    questions = quiz.get("questions", [])
    
    for i, question in enumerate(questions):
        st.markdown(f"### Question {i+1}")
        st.markdown(question.get("question"))
        
        # Display options
        options = question.get("options", [])
        selected_option = st.radio(
            f"Select answer for question {i+1}",
            options=options,
            key=f"q{i}",
            label_visibility="collapsed"
        )
        
        # Store selected answer
        if selected_option:
            # Extract the option letter (assumes "A. Option text" format)
            option_letter = selected_option.split(".")[0].strip()
            st.session_state.quiz_answers[str(i+1)] = option_letter
    
    # Submit button
    if st.button("Submit Quiz"):
        on_quiz_submit()

def render_quiz_results():
    """Render the results of the submitted quiz."""
    if not hasattr(st.session_state, "quiz_results"):
        st.error("No quiz results available.")
        st.session_state.show_quiz_results = False
        return
    
    results = st.session_state.quiz_results
    
    # Display score
    score = results.get("score", 0)
    total = results.get("total", 0)
    percentage = results.get("percentage", 0)
    passed = results.get("passed", False)
    
    st.subheader("Quiz Results")
    
    # Progress bar for score
    st.progress(percentage / 100, f"Score: {percentage}%")
    
    # Pass/fail message
    if passed:
        st.success(f"Congratulations! You passed with a score of {score}/{total} ({percentage}%).")
    else:
        st.error(f"You scored {score}/{total} ({percentage}%). Keep learning and try again!")
    
    # Feedback on individual questions
    st.subheader("Question Feedback")
    feedback = results.get("feedback", {})
    
    for question_id, is_correct in feedback.items():
        if is_correct:
            st.markdown(f"✅ Question {question_id}: Correct")
        else:
            st.markdown(f"❌ Question {question_id}: Incorrect")
    
    # Option to start a new quiz
    if st.button("Take Another Quiz"):
        st.session_state.show_quiz_results = False
        st.session_state.quiz_results = None
        st.session_state.current_quiz = None
        st.experimental_rerun()

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main content based on current page
    current_page = getattr(st.session_state, "current_page", "Chat")
    
    if current_page == "Chat":
        render_chat_ui()
    elif current_page == "Curriculum":
        render_curriculum_ui()
    elif current_page == "Module Details":
        render_module_details_ui()
    elif current_page == "Learning Path":
        render_learning_path_ui()
    elif current_page == "Progress":
        render_progress_ui()
    elif current_page == "Quiz":
        render_quiz_ui()

if __name__ == "__main__":
    main() 