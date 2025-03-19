# finance_teaching_workflow.py
"""
LangGraph implementation for Personal Finance Teaching Assistant
Coordinates the entire teaching flow using LangGraph's state management
"""

import os
from typing import Dict, List, Any, TypedDict, Annotated, Literal
from dotenv import load_dotenv
import json
from datetime import datetime

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import BaseTool, tool
from langchain_core.prompts import PromptTemplate

# Import the RAG implementation
from finance_rag import (
    retrieve_teaching_content,
    explain_financial_term,
    get_practical_examples,
    recommend_next_topic,
    get_user_level,
    supabase
)

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="claude-3-7-sonnet-20240229", temperature=0.2)

# Define the State Type
class FinanceTeachingState(TypedDict):
    """State for the finance teaching workflow."""
    messages: List[Dict]
    user_id: str
    user_level: int
    session_id: str
    current_module: str
    learning_goals: List[str]
    completed_modules: List[str]
    assessment_results: Dict[str, Any]
    next_steps: str
    topic_type: Literal["concept", "calculation", "strategy", "tool", "assessment", "general"]

# Define node functions

def analyze_query(state: FinanceTeachingState) -> Dict:
    """
    Analyze the user query to determine intent and topic type.
    This helps route to the right teaching approach.
    """
    # Get the latest user message
    user_messages = [m for m in state["messages"] if m.get("type") == "human"]
    if not user_messages:
        return state
    
    latest_message = user_messages[-1]["content"]
    
    # Analyze the query
    prompt = PromptTemplate.from_template(
        """You are analyzing a user's question about personal finance to determine how to best teach them.
        
        User query: {query}
        
        Classify this query into exactly ONE of these categories:
        - concept: Asking to understand a financial concept or term
        - calculation: Needs a financial calculation or formula
        - strategy: Asking about financial strategies or approaches
        - tool: Asking about financial tools or products
        - assessment: Wanting to test their knowledge
        - general: General question or doesn't fit other categories
        
        Respond with ONLY the category name, lowercase, no explanation.
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    topic_type = chain.invoke({"query": latest_message}).strip().lower()
    
    # Validate topic type
    valid_types = ["concept", "calculation", "strategy", "tool", "assessment", "general"]
    if topic_type not in valid_types:
        topic_type = "general"
    
    # Update state
    state["topic_type"] = topic_type
    
    return state

def retrieve_context(state: FinanceTeachingState) -> Dict:
    """
    Retrieve relevant context for the teaching response.
    Uses the RAG system to get pedagogically appropriate content.
    """
    # Get the latest user message
    user_messages = [m for m in state["messages"] if m.get("type") == "human"]
    if not user_messages:
        return state
    
    latest_message = user_messages[-1]["content"]
    
    # Get module context if available
    module_id = state.get("current_module")
    
    # Retrieve relevant content
    context = retrieve_teaching_content(
        latest_message, 
        state.get("user_id"), 
        module_id
    )
    
    # Add a system message with the context
    state["messages"].append({
        "type": "system",
        "content": f"Retrieved educational context: {context}"
    })
    
    return state

def explain_concept(state: FinanceTeachingState) -> Dict:
    """Handle explaining financial concepts."""
    # Get the latest user message
    user_messages = [m for m in state["messages"] if m.get("type") == "human"]
    if not user_messages:
        return state
    
    latest_message = user_messages[-1]["content"]
    
    # Extract key terms
    prompt = PromptTemplate.from_template(
        """Identify the main financial term or concept in this query.
        Return just the term, nothing else.
        
        Query: {query}
        """
    )
    chain = prompt | llm | StrOutputParser()
    term = chain.invoke({"query": latest_message}).strip()
    
    # Get explanation for the term
    explanation = explain_financial_term(term, state["user_level"])
    
    # Get a practical example
    example = get_practical_examples(term, 
        "beginner" if state["user_level"] <= 2 else 
        "intermediate" if state["user_level"] <= 4 else
        "advanced"
    )
    
    # Prepare teaching response
    prompt = ChatPromptTemplate.from_template(
        """You are a personal finance teacher explaining concepts to a student at level {level} (1-5).
        
        Concept to explain: {term}
        
        Technical explanation: {explanation}
        
        Practical example: {example}
        
        Create a clear, engaging explanation that:
        1. Starts with a simple definition anyone can understand
        2. Builds up to more details appropriate for level {level}
        3. Includes the practical example
        4. Ends with a single follow-up question to check understanding
        
        Keep your explanation conversational and encouraging.
        """
    )
    
    # Generate explanation
    chain = prompt | llm | StrOutputParser()
    teaching_response = chain.invoke({
        "level": state["user_level"],
        "term": term,
        "explanation": explanation,
        "example": example
    })
    
    # Add the response to messages
    state["messages"].append({
        "type": "ai",
        "content": teaching_response
    })
    
    return state

def teach_calculation(state: FinanceTeachingState) -> Dict:
    """Handle financial calculations and formulas."""
    # Get the latest user message
    user_messages = [m for m in state["messages"] if m.get("type") == "human"]
    if not user_messages:
        return state
    
    latest_message = user_messages[-1]["content"]
    
    # Get any relevant context already retrieved
    system_messages = [m for m in state["messages"] if m.get("type") == "system"]
    context = system_messages[-1]["content"] if system_messages else ""
    
    # Prepare teaching response for calculations
    prompt = ChatPromptTemplate.from_template(
        """You are a personal finance teacher helping with calculations at level {level} (1-5).
        
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
        """
    )
    
    # Generate explanation
    chain = prompt | llm | StrOutputParser()
    teaching_response = chain.invoke({
        "level": state["user_level"],
        "query": latest_message,
        "context": context
    })
    
    # Add the response to messages
    state["messages"].append({
        "type": "ai",
        "content": teaching_response
    })
    
    return state

def teach_strategy(state: FinanceTeachingState) -> Dict:
    """Handle financial strategies and approaches."""
    # Get the latest user message
    user_messages = [m for m in state["messages"] if m.get("type") == "human"]
    if not user_messages:
        return state
    
    latest_message = user_messages[-1]["content"]
    
    # Get any relevant context already retrieved
    system_messages = [m for m in state["messages"] if m.get("type") == "system"]
    context = system_messages[-1]["content"] if system_messages else ""
    
    # Prepare teaching response for strategies
    prompt = ChatPromptTemplate.from_template(
        """You are a personal finance teacher explaining strategies to a student at level {level} (1-5).
        
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
        """
    )
    
    # Generate explanation
    chain = prompt | llm | StrOutputParser()
    teaching_response = chain.invoke({
        "level": state["user_level"],
        "query": latest_message,
        "context": context
    })
    
    # Add the response to messages
    state["messages"].append({
        "type": "ai",
        "content": teaching_response
    })
    
    return state

def conduct_assessment(state: FinanceTeachingState) -> Dict:
    """Conduct knowledge assessment with quiz questions."""
    # Get the latest user message
    user_messages = [m for m in state["messages"] if m.get("type") == "human"]
    if not user_messages:
        return state
    
    latest_message = user_messages[-1]["content"]
    
    # Determine the topic for assessment
    prompt = PromptTemplate.from_template(
        """Identify the main financial topic the user wants to be assessed on.
        Return just the topic name, nothing else.
        
        User message: {message}
        """
    )
    chain = prompt | llm | StrOutputParser()
    topic = chain.invoke({"message": latest_message}).strip()
    
    # Create quiz questions
    assessment_prompt = PromptTemplate.from_template(
        """Create a mini-quiz to assess knowledge about {topic} at difficulty level {level} (1-5).
        
        Generate exactly 3 questions that:
        1. Test understanding rather than just facts
        2. Cover different aspects of {topic}
        3. Are appropriate for someone at knowledge level {level}
        
        For each question, provide:
        - The question itself
        - 4 multiple choice options (A, B, C, D)
        - The correct answer letter
        - A brief explanation of why it's correct
        
        Format your response as a clean, numbered list of questions with their options.
        Include a brief introduction and a closing note about what to do after completing the quiz.
        """
    )
    
    # Generate quiz
    chain = assessment_prompt | llm | StrOutputParser()
    quiz_content = chain.invoke({
        "topic": topic,
        "level": state["user_level"]
    })
    
    # Add the quiz to messages
    state["messages"].append({
        "type": "ai",
        "content": quiz_content
    })
    
    # Store quiz info for later
    state["assessment_results"] = {
        "topic": topic,
        "timestamp": datetime.now().isoformat(),
        "quiz_content": quiz_content
    }
    
    return state

def teach_about_tools(state: FinanceTeachingState) -> Dict:
    """Handle queries about financial tools and products."""
    # Get the latest user message
    user_messages = [m for m in state["messages"] if m.get("type") == "human"]
    if not user_messages:
        return state
    
    latest_message = user_messages[-1]["content"]
    
    # Get any relevant context already retrieved
    system_messages = [m for m in state["messages"] if m.get("type") == "system"]
    context = system_messages[-1]["content"] if system_messages else ""
    
    # Prepare teaching response for financial tools
    prompt = ChatPromptTemplate.from_template(
        """You are a personal finance teacher explaining financial tools and products to a student at level {level} (1-5).
        
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
        """
    )
    
    # Generate explanation
    chain = prompt | llm | StrOutputParser()
    teaching_response = chain.invoke({
        "level": state["user_level"],
        "query": latest_message,
        "context": context
    })
    
    # Add the response to messages
    state["messages"].append({
        "type": "ai",
        "content": teaching_response
    })
    
    return state

def handle_general_query(state: FinanceTeachingState) -> Dict:
    """Handle general financial questions that don't fit other categories."""
    # Get the latest user message
    user_messages = [m for m in state["messages"] if m.get("type") == "human"]
    if not user_messages:
        return state
    
    latest_message = user_messages[-1]["content"]
    
    # Get any relevant context already retrieved
    system_messages = [m for m in state["messages"] if m.get("type") == "system"]
    context = system_messages[-1]["content"] if system_messages else ""
    
    # Prepare response for general questions
    prompt = ChatPromptTemplate.from_template(
        """You are a personal finance teacher responding to a general question from a student at level {level} (1-5).
        
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
    )
    
    # Generate response
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "level": state["user_level"],
        "query": latest_message,
        "context": context
    })
    
    # Add the response to messages
    state["messages"].append({
        "type": "ai",
        "content": response
    })
    
    return state

def recommend_next_steps(state: FinanceTeachingState) -> Dict:
    """Add recommendations for what to learn next."""
    # Check if we need to add next steps (don't do this for every message)
    # Only do this occasionally to not overwhelm the user
    ai_messages = [m for m in state["messages"] if m.get("type") == "ai"]
    if len(ai_messages) % 3 != 0:  # Only on every 3rd AI message
        return state
    
    # Get user ID
    user_id = state.get("user_id")
    if not user_id:
        return state
    
    # Get recommendation
    recommendation = recommend_next_topic(user_id)
    
    # Format next steps message
    next_steps = f"""
    \n\n**Suggested Next Steps:**
    
    Based on your progress, I recommend learning about **{recommendation['title']}**.
    
    This module covers:
    - {', '.join(recommendation['topics'][:3])}
    
    {recommendation['reason']}
    
    Would you like to start learning about this topic?
    """
    
    # Update the most recent AI message to include the recommendation
    latest_ai_message = ai_messages[-1]
    latest_ai_message["content"] += next_steps
    
    # Store next steps in state
    state["next_steps"] = recommendation["module_id"]
    
    return state

def create_initial_state() -> FinanceTeachingState:
    """Create an initial state for the teaching workflow."""
    return {
        "messages": [],
        "user_id": None,
        "user_level": 1,  # Default to beginner level
        "session_id": str(datetime.now().timestamp()),
        "current_module": None,
        "learning_goals": [],
        "completed_modules": [],
        "assessment_results": {},
        "next_steps": "",
        "topic_type": "general"
    }

# Create the LangGraph
def build_finance_teaching_graph():
    """Build the LangGraph for finance teaching workflow."""
    # Initialize the graph
    workflow = StateGraph(FinanceTeachingState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("explain_concept", explain_concept)
    workflow.add_node("teach_calculation", teach_calculation)
    workflow.add_node("teach_strategy", teach_strategy)
    workflow.add_node("teach_about_tools", teach_about_tools)
    workflow.add_node("conduct_assessment", conduct_assessment)
    workflow.add_node("handle_general_query", handle_general_query)
    workflow.add_node("recommend_next_steps", recommend_next_steps)
    
    # Connect entry point to analysis
    workflow.set_entry_point("analyze_query")
    
    # Connect analysis to context retrieval
    workflow.add_edge("analyze_query", "retrieve_context")
    
    # Connect context retrieval to appropriate teaching method based on topic type
    workflow.add_conditional_edges(
        "retrieve_context",
        lambda state: state["topic_type"],
        {
            "concept": "explain_concept",
            "calculation": "teach_calculation",
            "strategy": "teach_strategy",
            "tool": "teach_about_tools",
            "assessment": "conduct_assessment",
            "general": "handle_general_query"
        }
    )
    
    # Connect all teaching methods to next steps recommendation
    workflow.add_edge("explain_concept", "recommend_next_steps")
    workflow.add_edge("teach_calculation", "recommend_next_steps")
    workflow.add_edge("teach_strategy", "recommend_next_steps")
    workflow.add_edge("teach_about_tools", "recommend_next_steps")
    workflow.add_edge("conduct_assessment", "recommend_next_steps")
    workflow.add_edge("handle_general_query", "recommend_next_steps")
    
    # End the workflow after recommending next steps
    workflow.add_edge("recommend_next_steps", END)
    
    # Compile the graph
    return workflow.compile()

# Function to process user message
def process_user_message(message: str, user_id: str = None, session_state: Dict = None) -> Dict:
    """
    Process a user message through the teaching workflow.
    
    Args:
        message: The user's message
        user_id: Optional user ID
        session_state: Optional existing session state
    
    Returns:
        Updated session state with AI response
    """
    # Get the graph
    graph = build_finance_teaching_graph()
    
    # Initialize or update state
    if session_state is None:
        state = create_initial_state()
        if user_id:
            state["user_id"] = user_id
            state["user_level"] = get_user_level(user_id)
    else:
        state = session_state
    
    # Add user message
    state["messages"].append({
        "type": "human",
        "content": message
    })
    
    # Process through graph
    result = graph.invoke(state)
    
    # Return updated state
    return result

# Example usage
if __name__ == "__main__":
    # Example conversation
    messages = [
        "Can you explain what compound interest is?",
        "How do I calculate how much I need for retirement?",
        "What's a good strategy for paying off debt?",
        "How do I open a Roth IRA?",
        "Test my knowledge about investing basics"
    ]
    
    # Process each message in sequence
    state = None
    for message in messages:
        print(f"\nUser: {message}\n")
        state = process_user_message(message, user_id="example_user", session_state=state)
        
        # Print the AI response
        ai_messages = [m for m in state["messages"] if m.get("type") == "ai"]
        if ai_messages:
            print(f"AI: {ai_messages[-1]['content']}\n")
            print("-" * 80)