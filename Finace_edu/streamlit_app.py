import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv

# Import our RAG and curriculum components
from finance_rag import (
    retrieve_teaching_content,
    explain_financial_term,
    get_practical_examples,
    recommend_next_topic,
    get_user_progress,
    get_user_level
)
from curriculum_roadmap import (
    CURRICULUM_ROADMAP,
    get_all_modules,
    get_module_by_id,
    get_modules_by_level,
    get_level_assessment
)

# Load environment variables
load_dotenv()

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_module" not in st.session_state:
        st.session_state.current_module = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = "test_user"  # For demo purposes
    if "knowledge_level" not in st.session_state:
        st.session_state.knowledge_level = 1
    if "current_quiz" not in st.session_state:
        st.session_state.current_quiz = None

def generate_module_quiz(module: dict):
    """Generate a quiz for a specific module."""
    if "current_quiz" not in st.session_state:
        st.session_state.current_quiz = None
    
    # Get assessment for the module's level
    level = int(module.get("level", 1))
    assessment = get_level_assessment(level)
    
    if not assessment:
        st.error("No assessment available for this module.")
        return
    
    # Generate quiz questions based on module content
    questions = []
    
    # Add questions from module content if available
    if "assessment_questions" in module:
        questions.extend(module["assessment_questions"])
    
    # Add questions from key concepts
    if "key_concepts" in module:
        for concept, details in module["key_concepts"].items():
            question = {
                "question": f"Which best describes {concept.replace('_', ' ')}?",
                "options": [
                    details.get("explanation", ""),
                    "Incorrect explanation 1",
                    "Incorrect explanation 2",
                    "Incorrect explanation 3"
                ],
                "correct_answer": details.get("explanation", ""),
                "explanation": f"The correct explanation includes {details.get('real_world_example', '')}"
            }
            questions.append(question)
    
    # If we still don't have enough questions, add some generic ones
    while len(questions) < 3:
        questions.append({
            "question": f"Sample question about {module['title']}?",
            "options": [
                "Correct answer",
                "Incorrect answer 1",
                "Incorrect answer 2",
                "Incorrect answer 3"
            ],
            "correct_answer": "Correct answer",
            "explanation": "This is a sample explanation."
        })
    
    # Store the quiz in session state
    st.session_state.current_quiz = {
        "module_id": module["module_id"],
        "questions": questions[:3],  # Limit to 3 questions
        "user_answers": {},
        "submitted": False
    }
    
    # Display the quiz
    display_quiz()

def display_quiz():
    """Display the current quiz and handle submissions."""
    if not st.session_state.current_quiz:
        return
    
    quiz = st.session_state.current_quiz
    
    st.subheader("📝 Module Quiz")
    
    # Display each question
    for i, question in enumerate(quiz["questions"]):
        st.write(f"\n**Question {i+1}:** {question['question']}")
        
        # Create a unique key for each question
        key = f"quiz_answer_{i}"
        
        # Get user's answer
        answer = st.radio(
            "Select your answer:",
            question["options"],
            key=key,
            index=None  # No default selection
        )
        
        # Store the answer
        if answer:
            quiz["user_answers"][i] = answer
    
    # Submit button
    if st.button("Submit Quiz"):
        submit_quiz()

def submit_quiz():
    """Handle quiz submission and show results."""
    quiz = st.session_state.current_quiz
    if not quiz or quiz["submitted"]:
        return
    
    # Calculate score
    correct_answers = 0
    total_questions = len(quiz["questions"])
    
    results = []
    for i, question in enumerate(quiz["questions"]):
        user_answer = quiz["user_answers"].get(i)
        is_correct = user_answer == question["correct_answer"]
        if is_correct:
            correct_answers += 1
        
        results.append({
            "question": question["question"],
            "user_answer": user_answer,
            "correct_answer": question["correct_answer"],
            "is_correct": is_correct,
            "explanation": question["explanation"]
        })
    
    # Calculate percentage
    score_percentage = (correct_answers / total_questions) * 100
    
    # Display results
    st.success(f"Quiz completed! Score: {score_percentage:.1f}%")
    
    for result in results:
        if result["is_correct"]:
            st.write("✅ " + result["question"])
        else:
            st.write("❌ " + result["question"])
            st.write("Your answer:", result["user_answer"])
            st.write("Correct answer:", result["correct_answer"])
            st.write("Explanation:", result["explanation"])
    
    # Mark quiz as submitted
    quiz["submitted"] = True
    
    # Update user progress
    update_progress(quiz["module_id"], score_percentage)

def update_progress(module_id: str, score: float):
    """Update user progress after completing a quiz."""
    try:
        # Get current progress
        progress = get_user_progress(st.session_state.user_id)
        
        # Update quiz scores
        quiz_scores = progress.get("quiz_scores", {})
        quiz_scores[module_id] = score
        
        # If score is passing (>= 70%), mark module as completed
        if score >= 70:
            completed_modules = progress.get("completed_modules", [])
            if module_id not in completed_modules:
                completed_modules.append(module_id)
                st.success(f"🎉 Congratulations! You've completed the {module_id} module!")
        
        # Here you would typically update the database
        # For now, we'll just show a success message
        st.success("Progress updated successfully!")
        
    except Exception as e:
        st.error(f"Error updating progress: {str(e)}")

def get_personalized_learning_path(user_id: str) -> list:
    """Generate a personalized learning path based on user's progress and interests."""
    user_progress = get_user_progress(user_id)
    user_level = user_progress.get("knowledge_level", 1)
    completed_modules = user_progress.get("completed_modules", [])
    
    # Get all available modules
    all_modules = get_all_modules()
    
    # Filter and sort modules based on user's level and progress
    learning_path = []
    for module in all_modules:
        module_level = int(module.get("level", 1))
        
        # Include modules that are:
        # 1. Not completed
        # 2. At user's current level or one level above/below
        if (module["module_id"] not in completed_modules and 
            abs(module_level - user_level) <= 1):
            
            # Add practical information to the module
            enhanced_module = {
                **module,
                "practical_examples": get_practical_examples(module["title"], 
                    "beginner" if user_level <= 2 else 
                    "intermediate" if user_level <= 4 else "advanced"),
                "key_terms": explain_financial_term(module["title"], user_level),
                "estimated_completion": module.get("estimated_time", "2-3 hours"),
                "prerequisites": get_prerequisites(module["module_id"]),
                "learning_outcomes": get_learning_outcomes(module)
            }
            learning_path.append(enhanced_module)
    
    return learning_path

def get_prerequisites(module_id: str) -> list:
    """Get prerequisites for a module."""
    # Map of module prerequisites
    prereq_map = {
        "module_1_1": [],  # No prerequisites for first module
        "module_1_2": ["module_1_1"],  # Need Money Basics before Budgeting
        "module_1_3": ["module_1_2"],  # Need Budgeting before Banking
        "module_2_1": ["module_1_1", "module_1_2"],  # Need basics before Emergency Fund
        # Add more mappings as needed
    }
    return prereq_map.get(module_id, [])

def get_learning_outcomes(module: dict) -> list:
    """Generate specific learning outcomes for a module."""
    outcomes = []
    for topic in module.get("topics", []):
        outcomes.append(f"Understand and apply {topic.lower()}")
    
    # Add practical outcomes
    if "exercises" in module:
        for exercise in module["exercises"]:
            outcomes.append(f"Complete practical exercise: {exercise}")
    
    return outcomes

def render_module_card(module: dict):
    """Render a module card with enhanced information."""
    with st.expander(f"📚 {module['title']} (Level {module['level']})"):
        st.write(f"**Description:** {module['description']}")
        
        # Show prerequisites if any
        prereqs = get_prerequisites(module['module_id'])
        if prereqs:
            st.write("**Prerequisites:**")
            for prereq in prereqs:
                st.write(f"- Complete {get_module_by_id(prereq)['title']}")
        
        # Show topics
        st.write("**Topics covered:**")
        for topic in module.get("topics", []):
            st.write(f"- {topic}")
        
        # Show practical examples
        if "practical_examples" in module:
            st.write("**Practical Application:**")
            st.write(module["practical_examples"])
        
        # Show exercises
        if "exercises" in module:
            st.write("**Exercises:**")
            for ex in module["exercises"]:
                st.write(f"- {ex}")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Module", key=f"start_{module['module_id']}"):
                st.session_state.current_module = module
        with col2:
            if st.button("Take Quiz", key=f"quiz_{module['module_id']}"):
                generate_module_quiz(module)

def render_learning_path():
    """Render the personalized learning path."""
    st.header("📚 Your Personal Finance Learning Journey")
    
    # Get user's learning path
    learning_path = get_personalized_learning_path(st.session_state.user_id)
    
    # Show current level and progress
    user_progress = get_user_progress(st.session_state.user_id)
    st.subheader(f"Current Level: {user_progress.get('knowledge_level', 1)}/5")
    
    # Progress bar
    completed = len(user_progress.get("completed_modules", []))
    total = len(get_all_modules())
    st.progress(completed / total, f"Completed {completed}/{total} modules")
    
    # Display modules in the learning path
    st.subheader("Recommended Modules")
    for module in learning_path:
        render_module_card(module)

def render_chat_interface():
    """Render the chat interface for asking questions."""
    st.header("💬 Ask Your Finance Questions")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about personal finance..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response using RAG
        response = retrieve_teaching_content(
            prompt,
            st.session_state.user_id,
            st.session_state.current_module["module_id"] if st.session_state.current_module else None
        )
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.set_page_config(
        page_title="Personal Finance Teacher",
        page_icon="💰",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Learning Path", "Chat", "Progress"])
    
    # Main content
    if page == "Learning Path":
        render_learning_path()
    elif page == "Chat":
        render_chat_interface()
    else:  # Progress
        st.header("📊 Your Progress")
        # Add progress tracking visualization here

if __name__ == "__main__":
    main() 