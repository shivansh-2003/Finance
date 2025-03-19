"""
PERSONAL FINANCE TEACHING ASSISTANT - MAIN APPLICATION

This is the main entry point for the Personal Finance Teaching Assistant application.
It integrates the curriculum roadmap, RAG system, and LangGraph workflow.
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our components
from curriculum_roadmap import (
    get_all_modules,
    get_module_by_id,
    get_modules_by_level,
    get_next_module,
    get_level_assessment
)

from finance_rag import (
    retrieve_teaching_content,
    explain_financial_term,
    get_practical_examples,
    recommend_next_topic,
    get_user_level,
    get_user_progress,
    update_user_progress,
    generate_learning_path
)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Personal Finance Teaching Assistant",
    description="An educational assistant to teach people about personal finance from basics to advanced concepts",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request/response models
class UserQuery(BaseModel):
    user_id: Optional[str] = None
    query: str
    session_id: Optional[str] = None

class ModuleRequest(BaseModel):
    user_id: str
    module_id: str

class QuizAnswer(BaseModel):
    user_id: str
    quiz_id: str
    answers: Dict[str, str]
    
class LearningPathRequest(BaseModel):
    user_id: Optional[str] = None
    goal: Optional[str] = None
    
class AssessmentRequest(BaseModel):
    user_id: str
    topic: str
    level: Optional[int] = None

# Create a global variable to store session states
# In production, use Redis or a database for session storage
session_states = {}

# Endpoints

@app.post("/chat")
async def chat_endpoint(request: UserQuery):
    """Process a user chat message and return an educational response."""
    
    # Get or create session state
    session_id = request.session_id or str(datetime.now().timestamp())
    
    if session_id in session_states:
        # Use existing session
        session_state = session_states[session_id]
    else:
        # Create new session with defaults
        user_level = get_user_level(request.user_id) if request.user_id else 1
        
        session_state = {
            "messages": [],
            "user_id": request.user_id,
            "user_level": user_level,
            "session_id": session_id,
            "current_module": None,
            "learning_goals": [],
            "completed_modules": [],
            "assessment_results": {},
            "next_steps": "",
            "topic_type": "general"
        }
    
    # Add the user's message to the state
    session_state["messages"].append({
        "type": "human",
        "content": request.query,
        "timestamp": datetime.now().isoformat()
    })
    
    # Process the message using the teaching workflow
    # In a real implementation, this would call the LangGraph process
    # For now, we'll use a simpler implementation
    
    # Step 1: Analyze query to determine intent
    query_type = analyze_query_type(request.query)
    
    # Step 2: Retrieve relevant context from RAG
    context = retrieve_teaching_content(
        request.query,
        request.user_id,
        session_state.get("current_module")
    )
    
    # Step 3: Generate appropriate response based on query type
    response = generate_response(
        query_type,
        request.query,
        context,
        session_state["user_level"]
    )
    
    # Step 4: Add response to session
    session_state["messages"].append({
        "type": "ai",
        "content": response,
        "timestamp": datetime.now().isoformat()
    })
    
    # Step 5: Save updated session state
    session_states[session_id] = session_state
    
    # Return response
    return {
        "response": response,
        "session_id": session_id,
        "user_level": session_state["user_level"]
    }

@app.get("/curriculum")
async def get_curriculum():
    """Get the full curriculum roadmap."""
    modules = get_all_modules()
    return {"modules": modules}

@app.get("/curriculum/modules/level/{level}")
async def get_level_modules(level: int):
    """Get modules for a specific level."""
    if level < 1 or level > 5:
        raise HTTPException(status_code=400, detail="Level must be between 1 and 5")
    
    modules = get_modules_by_level(level)
    return {"modules": modules}

@app.get("/curriculum/module/{module_id}")
async def get_module(module_id: str):
    """Get details for a specific module."""
    module = get_module_by_id(module_id)
    
    if not module:
        raise HTTPException(status_code=404, detail=f"Module {module_id} not found")
    
    return {"module": module}

@app.post("/learning/complete-module")
async def complete_module(request: ModuleRequest):
    """Mark a module as completed for a user."""
    # Update user progress
    update_user_progress(request.user_id, module_id=request.module_id)
    
    # Get next module recommendation
    next_module = get_next_module(request.module_id)
    
    return {
        "status": "success",
        "message": f"Module {request.module_id} marked as completed",
        "next_module": next_module
    }

@app.post("/assessment/quiz")
async def generate_quiz(request: AssessmentRequest):
    """Generate a quiz on a specific topic for a user."""
    # Get user level if not provided
    level = request.level
    if not level and request.user_id:
        level = get_user_level(request.user_id)
    
    # Default to level 1 if still not set
    level = level or 1
    
    # Generate quiz questions
    # In production, this would use a more sophisticated approach
    quiz = generate_assessment_quiz(request.topic, level)
    
    return {
        "quiz_id": f"quiz_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "topic": request.topic,
        "level": level,
        "questions": quiz["questions"]
    }

@app.post("/assessment/submit-quiz")
async def submit_quiz(request: QuizAnswer):
    """Submit quiz answers and get results."""
    # In production, fetch the correct answers from a database
    # For now, we'll assume a placeholder implementation
    
    results = evaluate_quiz_answers(request.quiz_id, request.answers)
    
    # Update user progress with quiz score
    update_user_progress(
        request.user_id,
        quiz_id=request.quiz_id,
        score=results["score"]
    )
    
    return results

@app.post("/learning/path")
async def get_learning_path(request: LearningPathRequest):
    """Get a personalized learning path for a user."""
    path = generate_learning_path(request.user_id, request.goal)
    
    return {"learning_path": path}

@app.get("/user/progress/{user_id}")
async def get_progress(user_id: str):
    """Get a user's learning progress."""
    progress = get_user_progress(user_id)
    
    return {"progress": progress}

# Helper functions

def analyze_query_type(query: str) -> str:
    """
    Determine the type of query being asked.
    In production, this would use the LLM-based classifier.
    
    Args:
        query: The user's query text
        
    Returns:
        Query type as a string
    """
    # For simplicity, using basic keyword matching
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["what is", "explain", "definition", "understand", "mean"]):
        return "concept"
    elif any(word in query_lower for word in ["calculate", "formula", "compute", "how much"]):
        return "calculation"
    elif any(word in query_lower for word in ["strategy", "approach", "how to", "steps", "plan"]):
        return "strategy"
    elif any(word in query_lower for word in ["tool", "app", "product", "service", "account"]):
        return "tool"
    elif any(word in query_lower for word in ["quiz", "test", "assess", "evaluate", "knowledge"]):
        return "assessment"
    else:
        return "general"

def generate_response(query_type: str, query: str, context: str, user_level: int) -> str:
    """
    Generate a response based on query type and context.
    In production, this would use the specialized LLM chains.
    
    Args:
        query_type: Type of the query
        query: The user's query text
        context: Retrieved context from RAG
        user_level: User's knowledge level (1-5)
        
    Returns:
        Formatted response
    """
    # In a real implementation, call the appropriate LLM chain based on query_type
    # For now, we're using a simplified version
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    llm = ChatOpenAI(model="claude-3-7-sonnet-20240229", temperature=0.2)
    
    # Different prompts for different query types
    prompts = {
        "concept": """You are a personal finance teacher explaining concepts to a student at level {level} (1-5).
            
            User query: {query}
            
            Retrieved context: {context}
            
            Create a clear, engaging explanation that:
            1. Starts with a simple definition anyone can understand
            2. Builds up to more details appropriate for level {level}
            3. Includes a practical example
            4. Ends with a single follow-up question to check understanding
            
            Keep your explanation conversational and encouraging.
            """,
            
        "calculation": """You are a personal finance teacher helping with calculations at level {level} (1-5).
            
            User query: {query}
            
            Retrieved context: {context}
            
            Create a clear explanation that:
            1. Identifies the financial calculation needed
            2. Shows the formula with variable names
            3. Explains each variable in the formula
            4. Provides a step-by-step example with real numbers
            5. If possible, gives a rule of thumb or simplified version
            
            If specific numbers are provided in the query, use those in your calculation.
            If not, use realistic example numbers.
            
            Keep your explanation practical and focused on what the calculation tells us about our finances.
            """,
            
        "strategy": """You are a personal finance teacher explaining strategies to a student at level {level} (1-5).
            
            User query: {query}
            
            Retrieved context: {context}
            
            Create a comprehensive yet accessible explanation that:
            1. Outlines the financial strategy or approach clearly
            2. Explains when and why this strategy is useful
            3. Provides step-by-step implementation guidance
            4. Mentions common pitfalls or misconceptions
            5. Includes specific action items the user can take
            
            For a level {level} student:
            - Levels 1-2: Focus on foundational strategies with minimal jargon
            - Levels 3-4: Include more nuanced considerations and trade-offs
            - Level 5: Discuss advanced optimization and edge cases
            
            Be encouraging and emphasize that good financial strategies are about consistency over time.
            """,
            
        "tool": """You are a personal finance teacher explaining financial tools and products to a student at level {level} (1-5).
            
            User query: {query}
            
            Retrieved context: {context}
            
            Create an informative explanation that:
            1. Clearly describes what the financial tool/product is
            2. Explains how it works and its main features
            3. Discusses the pros and cons
            4. Provides guidance on when and how to use it
            5. Includes considerations for choosing between alternatives
            
            For a level {level} student:
            - Levels 1-2: Focus on the basics with minimal technical details
            - Levels 3-4: Include more details about optimal usage and selection criteria
            - Level 5: Discuss advanced features, edge cases, and optimization
            
            Be objective and balanced in your assessment of financial products and tools.
            """,
            
        "assessment": """You are creating a mini-quiz to assess knowledge about personal finance for a student at level {level} (1-5).
            
            User query: {query}
            
            Retrieved context: {context}
            
            Create a mini-quiz with 3 multiple-choice questions that:
            1. Tests understanding of relevant personal finance concepts
            2. Covers different aspects of the topic
            3. Is appropriate for someone at knowledge level {level}
            
            Format your response as an intro paragraph followed by 3 numbered questions with A, B, C, D options.
            Include a closing note about what to do after completing the quiz.
            """,
            
        "general": """You are a personal finance teacher responding to a general question from a student at level {level} (1-5).
            
            User query: {query}
            
            Retrieved context: {context}
            
            Create a helpful response that:
            1. Directly addresses their question in a clear, friendly way
            2. Provides accurate and relevant financial information
            3. Includes practical advice they can apply
            4. Suggests related topics they might want to learn about next
            
            Match the detail level to their knowledge level ({level}/5).
            Keep your response conversational but educational.
            """
    }
    
    # Use the appropriate prompt for the query type
    prompt_template = prompts.get(query_type, prompts["general"])
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Create the chain
    chain = prompt | llm | StrOutputParser()
    
    # Generate the response
    response = chain.invoke({
        "level": user_level,
        "query": query,
        "context": context
    })
    
    return response

def generate_assessment_quiz(topic: str, level: int) -> Dict[str, Any]:
    """
    Generate a quiz for a specific topic and difficulty level.
    In production, this would use a more sophisticated approach.
    
    Args:
        topic: Topic to generate quiz for
        level: Difficulty level (1-5)
        
    Returns:
        Dictionary with quiz questions
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    import json
    
    llm = ChatOpenAI(model="claude-3-7-sonnet-20240229", temperature=0.2)
    
    # Prompt for quiz generation
    prompt = PromptTemplate.from_template(
        """Generate a quiz to assess knowledge about {topic} at difficulty level {level} (1-5).
        
        Create exactly 3 multiple-choice questions that:
        1. Test understanding rather than just facts
        2. Cover different aspects of {topic}
        3. Are appropriate for someone at knowledge level {level}
        
        For each question, provide:
        - "question": The question text
        - "options": An array of 4 options (A, B, C, D format)
        - "correct_answer": The letter of the correct option
        - "explanation": A brief explanation of why it's correct
        
        Return your response as a valid JSON object with a "questions" array containing these question objects.
        """
    )
    
    # Generate quiz content
    response = llm.invoke(prompt.format(topic=topic, level=level))
    
    # Extract JSON from the response
    # In a real system, we'd use a more robust method, possibly with output parsing
    try:
        # Try to parse the whole response as JSON
        quiz_data = json.loads(response.content)
    except:
        # If that fails, try to find JSON in the text
        import re
        json_match = re.search(r'```json\n(.*?)```', response.content, re.DOTALL)
        if json_match:
            try:
                quiz_data = json.loads(json_match.group(1))
            except:
                # Fallback to a simple quiz
                quiz_data = {
                    "questions": [
                        {
                            "question": f"This is a placeholder question about {topic}?",
                            "options": [
                                "A. Option A",
                                "B. Option B",
                                "C. Option C",
                                "D. Option D"
                            ],
                            "correct_answer": "A",
                            "explanation": "This is a placeholder explanation."
                        }
                    ]
                }
        else:
            # Fallback to a simple quiz
            quiz_data = {
                "questions": [
                    {
                        "question": f"This is a placeholder question about {topic}?",
                        "options": [
                            "A. Option A",
                            "B. Option B",
                            "C. Option C",
                            "D. Option D"
                        ],
                        "correct_answer": "A",
                        "explanation": "This is a placeholder explanation."
                    }
                ]
            }
    
    return quiz_data

def evaluate_quiz_answers(quiz_id: str, answers: Dict[str, str]) -> Dict[str, Any]:
    """
    Evaluate user answers for a quiz.
    In production, this would fetch the correct answers from a database.
    
    Args:
        quiz_id: The ID of the quiz
        answers: Dictionary of user answers (question_id -> answer)
        
    Returns:
        Dictionary with evaluation results
    """
    # For this implementation, we'll use a placeholder approach
    # In production, fetch the correct answers from the database
    
    # Placeholder correct answers (would come from database)
    correct_answers = {
        "1": "A",
        "2": "C",
        "3": "B"
    }
    
    # Score the quiz
    score = 0
    feedback = {}
    
    for question_id, user_answer in answers.items():
        if question_id in correct_answers:
            is_correct = user_answer == correct_answers[question_id]
            feedback[question_id] = is_correct
            if is_correct:
                score += 1
    
    # Calculate percentage
    total_questions = len(correct_answers)
    percentage = int((score / total_questions) * 100) if total_questions > 0 else 0
    
    return {
        "score": score,
        "total": total_questions,
        "percentage": percentage,
        "feedback": feedback,
        "passed": percentage >= 70  # Assuming 70% is passing
    }

# Start the server if run directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 