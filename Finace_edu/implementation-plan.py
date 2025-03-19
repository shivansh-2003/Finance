"""
PERSONAL FINANCE TEACHING AGENT IMPLEMENTATION PLAN

This document outlines the complete implementation approach for building the 
personal finance teaching assistant with the specified tech stack.
"""

# 1. DATA PREPARATION AND INGESTION
"""
Step 1: Prepare high-quality finance content for each curriculum module
- Create or curate content for each week in the curriculum
- Organize content into clear modules with examples and exercises
- Include quiz questions and practical exercises for each topic
"""

# Example structure for content preparation
content_structure = {
    "module_id": "week_1_money_basics",
    "title": "Money Basics",
    "level": 1,
    "content_sections": [
        {
            "title": "Understanding Income and Expenses",
            "content": "Detailed explanation text...",
            "examples": ["Example 1...", "Example 2..."],
            "exercises": ["Exercise 1...", "Exercise 2..."]
        },
        # Additional sections
    ],
    "quiz_questions": [
        {
            "question": "What's the difference between gross income and net income?",
            "options": ["Option A...", "Option B...", "Option C...", "Option D..."],
            "correct_answer": "Option B",
            "explanation": "Detailed explanation of answer..."
        },
        # Additional questions
    ],
    "next_steps": "In the next module, we'll learn about budgeting fundamentals..."
}

# 2. DATABASE SETUP
"""
Step 2: Set up Supabase database with the following tables:
- documents: Vector store for curriculum content
- user_progress: Track individual user progress
- assessments: Store quiz questions and answers
- feedback: Collect user feedback and questions
"""

# Example database schema (SQL)
"""
-- Vector store table (already implemented in your code)
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  content TEXT,
  metadata JSONB,
  embedding VECTOR(1536)
);

-- User progress tracking
CREATE TABLE user_progress (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users(id),
  module_id TEXT,
  completed BOOLEAN DEFAULT FALSE,
  quiz_score INTEGER,
  last_interaction TIMESTAMP WITH TIME ZONE,
  notes TEXT
);

-- Feedback collection
CREATE TABLE feedback (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users(id),
  module_id TEXT,
  question TEXT,
  sentiment TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

# 3. ENHANCED DOCUMENT INGESTION
"""
Step 3: Enhance the document ingestion process to support curriculum structure
"""

# Enhanced document ingestion script
def ingest_curriculum_content():
    """
    Enhanced version of your current ingest_in_db.py script
    that handles curriculum metadata and structure
    """
    # import basics
    import os
    from dotenv import load_dotenv
    import json
    from pathlib import Path

    # import langchain components
    from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import SupabaseVectorStore
    from langchain_openai import OpenAIEmbeddings

    # import supabase
    from supabase.client import Client, create_client

    # load environment variables
    load_dotenv()  

    # initiate supabase db
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)

    # initiate embeddings model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Process curriculum content
    curriculum_dir = Path("curriculum")
    
    # Process each module
    all_docs = []
    for module_dir in curriculum_dir.iterdir():
        if module_dir.is_dir():
            # Load module metadata
            with open(module_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
            
            # Process content files
            for content_file in module_dir.glob("*.md"):
                loader = TextLoader(str(content_file))
                docs = loader.load()
                
                # Add metadata to each document
                for doc in docs:
                    doc.metadata.update({
                        "module_id": metadata["module_id"],
                        "level": metadata["level"],
                        "section": content_file.stem,
                        "topic": metadata["title"]
                    })
                    all_docs.append(doc)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)
    
    # Store in vector database with enhanced metadata
    vector_store = SupabaseVectorStore.from_documents(
        chunks,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents"
    )
    
    return f"Ingested {len(chunks)} curriculum chunks into the database"

# 4. AGENT IMPLEMENTATION
"""
Step 4: Implement specialized agents using LangChain and LangGraph
"""

# 4.1 Curriculum Agent - Helps navigate the learning path
def create_curriculum_agent():
    from langchain.agents import AgentExecutor
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_tool_calling_agent
    from langchain_core.prompts import PromptTemplate
    
    # Curriculum navigation tool
    @tool
    def get_curriculum_roadmap(user_level=None):
        """Retrieve the curriculum roadmap, optionally filtered by level."""
        # Implementation to fetch and return curriculum roadmap
        pass
    
    @tool
    def get_module_details(module_id):
        """Get detailed information about a specific module."""
        # Implementation to fetch module details from database
        pass
    
    @tool
    def recommend_next_module(user_id):
        """Recommend the next module based on user progress."""
        # Implementation to analyze user progress and recommend next steps
        pass
    
    # Curriculum agent prompt
    prompt = PromptTemplate.from_template(
        """You are a curriculum advisor specializing in personal finance education.
        Your task is to help users navigate the personal finance curriculum.
        The curriculum is structured into 5 levels, from basics to advanced topics.
        
        Current user query: {input}
        
        If the user is asking about what to learn next, recommend appropriate modules.
        If the user is asking about specific topics, provide guidance on relevant modules.
        
        Use your tools to access curriculum information and make recommendations.
        
        {agent_scratchpad}
        """
    )
    
    # Create agent with tools
    tools = [get_curriculum_roadmap, get_module_details, recommend_next_module]
    llm = ChatOpenAI(model="claude-3-7-sonnet-20240229", temperature=0)
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4.2 Teaching Agent - Delivers educational content
def create_teaching_agent():
    from langchain.agents import AgentExecutor
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_tool_calling_agent
    from langchain_core.prompts import PromptTemplate
    
    # Enhanced RAG tool with teaching focus
    @tool
    def retrieve_educational_content(query, module_id=None):
        """
        Retrieve educational content relevant to the query.
        Optionally filter by module_id for specific module content.
        """
        # Implementation similar to your current retrieve function
        # but enhanced with pedagogical metadata
        pass
    
    @tool
    def get_examples_and_exercises(concept, difficulty="beginner"):
        """Get examples and exercises for a specific finance concept."""
        # Implementation to fetch relevant examples and exercises
        pass
    
    @tool 
    def explain_financial_term(term):
        """Provide a clear explanation of a financial term or concept."""
        # Implementation to fetch term definitions and explanations
        pass
    
    # Teaching agent prompt
    prompt = PromptTemplate.from_template(
        """You are a personal finance teacher with expertise in explaining complex concepts in simple terms.
        Your goal is to help users learn personal finance concepts step by step.
        
        User query: {input}
        
        When teaching:
        1. Start with fundamentals before moving to complex topics
        2. Use clear explanations with real-world examples
        3. Provide actionable steps users can take
        4. Connect new concepts to previously learned material
        5. Check for understanding with simple questions
        
        Use your tools to retrieve relevant educational content.
        
        {agent_scratchpad}
        """
    )
    
    # Create agent with tools
    tools = [retrieve_educational_content, get_examples_and_exercises, explain_financial_term]
    llm = ChatOpenAI(model="claude-3-7-sonnet-20240229", temperature=0.2)
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4.3 Assessment Agent - Quizzes and evaluates understanding
def create_assessment_agent():
    from langchain.agents import AgentExecutor
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_tool_calling_agent
    from langchain_core.prompts import PromptTemplate
    
    @tool
    def generate_quiz(module_id, difficulty="medium"):
        """Generate a quiz for the specified module."""
        # Implementation to create relevant quiz questions
        pass
    
    @tool
    def evaluate_answer(question_id, user_answer):
        """Evaluate a user's answer to a quiz question."""
        # Implementation to assess answer correctness
        pass
    
    @tool
    def record_quiz_results(user_id, module_id, score):
        """Record quiz results in the user's progress tracker."""
        # Implementation to update user progress database
        pass
    
    # Assessment agent prompt
    prompt = PromptTemplate.from_template(
        """You are an assessment specialist focused on personal finance education.
        Your role is to help users test their understanding of financial concepts.
        
        User query: {input}
        
        For assessments:
        1. Create relevant questions that test understanding, not just memorization
        2. Provide constructive feedback on user answers
        3. Identify knowledge gaps and suggest review areas
        4. Celebrate correct answers and progress
        
        Use your tools to generate quizzes and evaluate understanding.
        
        {agent_scratchpad}
        """
    )
    
    # Create agent with tools
    tools = [generate_quiz, evaluate_answer, record_quiz_results]
    llm = ChatOpenAI(model="claude-3-7-sonnet-20240229", temperature=0.1)
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. LANGGRAPH ORCHESTRATION
"""
Step 5: Use LangGraph to orchestrate the agent workflow
"""

def create_finance_teaching_graph():
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, List, Dict, Any
    
    # Define the state
    class AgentState(TypedDict):
        user_input: str
        user_id: str
        context: Dict[str, Any]
        agent_outputs: Dict[str, Any]
        final_response: str
    
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Node functions
    def route_request(state):
        """Route the request to the appropriate agent."""
        user_input = state["user_input"].lower()
        
        if any(word in user_input for word in ["curriculum", "roadmap", "what should i learn", "next topic"]):
            return "curriculum_agent"
        elif any(word in user_input for word in ["quiz", "test", "assessment", "understand"]):
            return "assessment_agent"
        else:
            return "teaching_agent"
    
    def run_curriculum_agent(state):
        """Run the curriculum agent and update state."""
        curriculum_agent = create_curriculum_agent()
        response = curriculum_agent.invoke({"input": state["user_input"]})
        state["agent_outputs"]["curriculum"] = response
        return state
    
    def run_teaching_agent(state):
        """Run the teaching agent and update state."""
        teaching_agent = create_teaching_agent()
        response = teaching_agent.invoke({
            "input": state["user_input"],
            "context": state["context"]
        })
        state["agent_outputs"]["teaching"] = response
        return state
    
    def run_assessment_agent(state):
        """Run the assessment agent and update state."""
        assessment_agent = create_assessment_agent()
        response = assessment_agent.invoke({
            "input": state["user_input"],
            "user_id": state["user_id"]
        })
        state["agent_outputs"]["assessment"] = response
        return state
    
    def generate_response(state):
        """Format the final response based on agent outputs."""
        # Logic to create cohesive response from agent outputs
        if "curriculum" in state["agent_outputs"]:
            state["final_response"] = state["agent_outputs"]["curriculum"]["output"]
        elif "assessment" in state["agent_outputs"]:
            state["final_response"] = state["agent_outputs"]["assessment"]["output"]
        else:
            state["final_response"] = state["agent_outputs"]["teaching"]["output"]
        
        return state
    
    # Add nodes to graph
    workflow.add_node("router", route_request)
    workflow.add_node("curriculum_agent", run_curriculum_agent)
    workflow.add_node("teaching_agent", run_teaching_agent)
    workflow.add_node("assessment_agent", run_assessment_agent)
    workflow.add_node("response_generator", generate_response)
    
    # Add edges
    workflow.add_conditional_edges(
        "router",
        {
            "curriculum_agent": lambda x: x == "curriculum_agent",
            "teaching_agent": lambda x: x == "teaching_agent",
            "assessment_agent": lambda x: x == "assessment_agent",
        }
    )
    workflow.add_edge("curriculum_agent", "response_generator")
    workflow.add_edge("teaching_agent", "response_generator")
    workflow.add_edge("assessment_agent", "response_generator")
    workflow.add_edge("response_generator", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app

# 6. MAIN APPLICATION
"""
Step 6: Create the main application that users interact with
"""

def create_finance_teaching_assistant():
    """Create the main finance teaching assistant application."""
    from fastapi import FastAPI, Request
    from pydantic import BaseModel
    import uvicorn
    
    app = FastAPI(title="Personal Finance Teaching Assistant")
    finance_graph = create_finance_teaching_graph()
    
    class UserQuery(BaseModel):
        user_id: str
        query: str
    
    @app.post("/chat")
    async def chat_endpoint(request: UserQuery):
        """Process a user query and return educational response."""
        # Initialize state
        state = {
            "user_input": request.query,
            "user_id": request.user_id,
            "context": get_user_context(request.user_id),
            "agent_outputs": {},
            "final_response": ""
        }
        
        # Run the graph
        result = finance_graph.invoke(state)
        
        # Update user context and track progress
        update_user_context(request.user_id, request.query, result["final_response"])
        
        return {"response": result["final_response"]}
    
    def get_user_context(user_id):
        """Retrieve user context including progress and history."""
        # Implementation to fetch user data from Supabase
        pass
    
    def update_user_context(user_id, query, response):
        """Update user context after an interaction."""
        # Implementation to update user progress in Supabase
        pass
    
    return app

# 7. PROGRESS TRACKING SYSTEM
"""
Step 7: Implement a progress tracking system to provide personalized learning
"""

def create_progress_tracker():
    """Create a system to track user progress through the curriculum."""
    
    def get_user_progress(user_id):
        """Get the current progress for a user."""
        # Implementation to query user_progress table
        pass
    
    def update_user_progress(user_id, module_id, completed=False, quiz_score=None):
        """Update a user's progress on a module."""
        # Implementation to update user_progress table
        pass
    
    def generate_progress_report(user_id):
        """Generate a comprehensive progress report for a user."""
        # Implementation to create a summary of user's learning journey
        pass
    
    def recommend_next_steps(user_id):
        """Recommend next learning steps based on progress."""
        # Implementation to analyze progress and make recommendations
        pass
    
    return {
        "get_user_progress": get_user_progress,
        "update_user_progress": update_user_progress,
        "generate_progress_report": generate_progress_report,
        "recommend_next_steps": recommend_next_steps
    }

# 8. PUTTING IT ALL TOGETHER
"""
Step 8: Final implementation steps and deployment considerations
"""

def main():
    """Main function to run the application."""
    # 1. Ensure database is set up correctly
    # setup_database()
    
    # 2. Ingest curriculum content
    # ingest_curriculum_content()
    
    # 3. Create and start the application
    app = create_finance_teaching_assistant()
    
    # 4. Start the server (development mode)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
