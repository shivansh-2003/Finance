"""
FINANCE EDUCATION RAG IMPLEMENTATION

This module implements the Retrieval Augmented Generation (RAG) pipeline
for the personal finance teaching assistant.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Supabase
from supabase.client import Client, create_client

# Import the curriculum roadmap
from curriculum_roadmap import get_module_by_id, get_next_module, get_level_assessment

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# Initialize LLM with a temperature appropriate for teaching
llm = ChatOpenAI(model="claude-3-7-sonnet-20240229", temperature=0.2)

def retrieve_teaching_content(query: str, user_id: Optional[str] = None, module_id: Optional[str] = None) -> str:
    """
    Retrieve educational content relevant to a user query.
    
    Args:
        query: User's question or topic
        user_id: Optional user ID to personalize results
        module_id: Optional module ID to filter content
    
    Returns:
        Relevant educational content as a string
    """
    # Create metadata filters if module ID is provided
    metadata_filters = {}
    if module_id:
        module = get_module_by_id(module_id)
        if module:
            metadata_filters = {"module_id": module_id}
    
    # Get user level if user_id is provided
    user_level = get_user_level(user_id) if user_id else 1
    
    # Create search query enhancer
    enhancer_prompt = PromptTemplate.from_template(
        """You are helping a personal finance teaching system retrieve educational content.
        
        User query: {query}
        
        Your task is to enhance this query to improve retrieval of personal finance educational content.
        Add relevant financial terms and concepts related to the query.
        If the query is vague, make it more specific to personal finance education.
        The user's knowledge level is {level} (1-5, where 1 is beginner and 5 is expert).
        
        Enhanced query:"""
    )
    
    enhancer_chain = enhancer_prompt | llm | StrOutputParser()
    enhanced_query = enhancer_chain.invoke({"query": query, "level": user_level})
    
    # Retrieve documents
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "filter": metadata_filters}
    )
    
    docs = retriever.get_relevant_documents(enhanced_query)
    
    # Process the retrieved documents
    content_parts = []
    for doc in docs:
        # Add metadata information about the source
        source_info = f"Source: {doc.metadata.get('title', 'Unknown Source')}"
        if "module_id" in doc.metadata:
            source_info += f" - Module: {doc.metadata.get('module_id')}"
        if "section" in doc.metadata:
            source_info += f" - Section: {doc.metadata.get('section')}"
        
        # Add the content with its source
        content_parts.append(f"{source_info}\n\n{doc.page_content}")
    
    # If no results, return a fallback message
    if not content_parts:
        return "I don't have specific curriculum content on this topic yet. Here's some general guidance based on financial best practices."
    
    # Join all content parts
    return "\n\n---\n\n".join(content_parts)

def explain_financial_term(term: str, user_level: int = 1) -> str:
    """
    Get an explanation for a financial term, adjusted for the user's level.
    
    Args:
        term: The financial term to explain
        user_level: User's knowledge level (1-5)
    
    Returns:
        An explanation of the term
    """
    # First try to get the term from our vector store
    query = f"Define and explain the financial term: {term}"
    
    # Determine the appropriate level of explanation
    level_desc = "beginner"
    if user_level > 3:
        level_desc = "advanced"
    elif user_level > 1:
        level_desc = "intermediate"
    
    # Enhanced query for better retrieval
    enhanced_query = f"Definition explanation {term} financial term {level_desc} level"
    
    # Retrieve definition
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents(enhanced_query)
    
    # If we have results from the vector store, process them
    if docs:
        content_parts = [doc.page_content for doc in docs]
        retrieved_content = "\n\n".join(content_parts)
        
        # Format the explanation using the LLM to make it appropriate for the user level
        prompt = PromptTemplate.from_template(
            """You are explaining the financial term '{term}' to someone at knowledge level {level} (1-5).
            
            Retrieved information about this term:
            {content}
            
            Create a clear, concise explanation that:
            1. Starts with a 1-sentence simple definition
            2. Expands with details appropriate for level {level}
            3. Includes any key points from the retrieved information
            4. Uses examples where helpful
            
            If the retrieved information is not relevant, create an accurate explanation based on your own knowledge.
            
            Keep your tone friendly and educational without being condescending.
            The explanation should be no more than 3-4 paragraphs.
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "term": term,
            "level": user_level,
            "content": retrieved_content
        })
    
    # Fallback to generating a explanation with the LLM if no content found
    prompt = PromptTemplate.from_template(
        """You are a personal finance educator. Explain the financial term '{term}' to a {level_desc} level student.
        
        Your explanation should:
        1. Start with a clear, concise definition
        2. Explain why this term is important in personal finance
        3. Give an example of how it applies in real life
        4. Be appropriate for a {level_desc} level understanding
        
        Keep your explanation accurate, educational and concise (3-4 paragraphs maximum).
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "term": term,
        "level_desc": level_desc
    })

def get_practical_examples(concept: str, difficulty: str = "beginner") -> str:
    """
    Get practical examples for a financial concept.
    
    Args:
        concept: The financial concept
        difficulty: Difficulty level (beginner, intermediate, advanced)
    
    Returns:
        Practical examples as a string
    """
    # Try to retrieve examples from the vector store
    query = f"practical examples {concept} {difficulty} personal finance"
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents(query)
    
    # If we have results from the vector store, process them
    if docs:
        content_parts = [doc.page_content for doc in docs]
        retrieved_content = "\n\n".join(content_parts)
        
        # Format the examples using the LLM
        prompt = PromptTemplate.from_template(
            """You are providing practical examples of '{concept}' at a {difficulty} level.
            
            Retrieved information:
            {content}
            
            Create 2-3 clear, practical examples that:
            1. Illustrate how {concept} applies in real-life financial situations
            2. Are appropriate for {difficulty} level understanding
            3. Include specific numbers and scenarios
            4. Highlight the key principles in action
            
            If the retrieved information doesn't contain good examples, create your own accurate examples.
            
            Format as a numbered list of examples, each with a brief scenario and explanation.
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "concept": concept,
            "difficulty": difficulty,
            "content": retrieved_content
        })
    
    # Fallback to generating examples with the LLM if no content found
    prompt = PromptTemplate.from_template(
        """You are a personal finance educator. Provide 2-3 practical examples of '{concept}' for a {difficulty} level student.
        
        Each example should:
        1. Be a realistic scenario involving {concept}
        2. Include specific numbers and calculations if relevant
        3. Show the practical application in everyday life
        4. Highlight why this concept matters
        
        Format as a numbered list. Make your examples clear, practical, and appropriate for {difficulty} level understanding.
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "concept": concept,
        "difficulty": difficulty
    })

def recommend_next_topic(user_id: str) -> Dict[str, Any]:
    """
    Recommend what to study next based on user's progress.
    
    Args:
        user_id: The user's ID
    
    Returns:
        Dictionary with recommendation details
    """
    # Get user's progress data
    user_progress = get_user_progress(user_id)
    user_level = get_user_level(user_id)
    
    # If user has completed modules, recommend the next one
    last_completed_module = user_progress.get("last_completed_module")
    if last_completed_module:
        next_module = get_next_module(last_completed_module)
        if next_module:
            return {
                "module_id": next_module["module_id"],
                "title": next_module["title"],
                "description": next_module["description"],
                "topics": next_module["topics"],
                "reason": "This is the next module in the curriculum sequence."
            }
    
    # If no completed modules or next module not found, recommend starting point
    # based on user's knowledge level
    level = max(1, min(5, user_level))
    modules = get_modules_by_level(level)
    
    if modules:
        # Recommend the first module of the appropriate level
        module = modules[0]
        return {
            "module_id": module["module_id"],
            "title": module["title"],
            "description": module["description"],
            "topics": module["topics"],
            "reason": f"This module is appropriate for your current knowledge level ({level}/5)."
        }
    
    # Fallback recommendation
    return {
        "module_id": "module_1_1",
        "title": "Money Basics",
        "description": "Understanding the fundamental concepts of money and personal finance",
        "topics": ["What is money and its functions", "Income vs. expenses", "Financial goals and values", "Basic financial terminology"],
        "reason": "It's best to start with the fundamentals of personal finance."
    }

def get_user_level(user_id: Optional[str]) -> int:
    """
    Get the user's knowledge level from their profile.
    
    Args:
        user_id: The user's ID
    
    Returns:
        Knowledge level (1-5)
    """
    if not user_id:
        return 1  # Default to beginner level
    
    try:
        # Fetch user progress data from Supabase
        response = supabase.table("user_progress").select("knowledge_level").eq("user_id", user_id).execute()
        
        # Check if we got results
        if response.data and len(response.data) > 0:
            return response.data[0].get("knowledge_level", 1)
        
        return 1  # Default to beginner if user not found
        
    except Exception as e:
        print(f"Error fetching user level: {e}")
        return 1  # Default to beginner level on error

def get_user_progress(user_id: str) -> Dict[str, Any]:
    """
    Get a user's learning progress details.
    
    Args:
        user_id: The user's ID
    
    Returns:
        Dictionary with user progress details
    """
    try:
        # Fetch user progress data
        response = supabase.table("user_progress").select("*").eq("user_id", user_id).execute()
        
        # Check if we got results
        if response.data and len(response.data) > 0:
            return response.data[0]
        
        # Default empty progress
        return {
            "completed_modules": [],
            "last_completed_module": None,
            "knowledge_level": 1,
            "quiz_scores": {}
        }
        
    except Exception as e:
        print(f"Error fetching user progress: {e}")
        return {
            "completed_modules": [],
            "last_completed_module": None,
            "knowledge_level": 1,
            "quiz_scores": {}
        }

def update_user_progress(user_id: str, module_id: Optional[str] = None, 
                         quiz_id: Optional[str] = None, score: Optional[int] = None) -> None:
    """
    Update a user's learning progress.
    
    Args:
        user_id: The user's ID
        module_id: Optional module ID that was completed
        quiz_id: Optional quiz ID that was completed
        score: Optional score from a quiz
    """
    try:
        # Get current progress
        current_progress = get_user_progress(user_id)
        
        # Update completed modules if provided
        if module_id:
            completed_modules = current_progress.get("completed_modules", [])
            if module_id not in completed_modules:
                completed_modules.append(module_id)
                
            # Update data to send to database
            update_data = {
                "completed_modules": completed_modules,
                "last_completed_module": module_id,
                "last_interaction": "now()"
            }
            
            # Update the progress in Supabase
            supabase.table("user_progress").upsert({
                "user_id": user_id,
                **update_data
            }).execute()
        
        # Update quiz scores if provided
        if quiz_id and score is not None:
            quiz_scores = current_progress.get("quiz_scores", {})
            quiz_scores[quiz_id] = score
            
            # Check if this is a level assessment
            if quiz_id.startswith("quiz_level_"):
                level = int(quiz_id.split("_")[-1])
                assessment = get_level_assessment(level)
                
                # If they passed, potentially update knowledge level
                if assessment and score >= assessment["passing_score"]:
                    # Only increase level if they passed a level above their current one
                    current_level = current_progress.get("knowledge_level", 1)
                    if level > current_level:
                        supabase.table("user_progress").upsert({
                            "user_id": user_id,
                            "knowledge_level": level,
                            "quiz_scores": quiz_scores,
                            "last_interaction": "now()"
                        }).execute()
                        return
            
            # Standard quiz score update
            supabase.table("user_progress").upsert({
                "user_id": user_id,
                "quiz_scores": quiz_scores,
                "last_interaction": "now()"
            }).execute()
            
    except Exception as e:
        print(f"Error updating user progress: {e}")

def generate_learning_path(user_id: Optional[str] = None, goal: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Generate a personalized learning path for a user.
    
    Args:
        user_id: The user's ID
        goal: Optional specific financial goal
    
    Returns:
        List of recommended modules in sequence
    """
    # Get user's knowledge level
    user_level = get_user_level(user_id) if user_id else 1
    
    # If a specific goal is provided, customize the path
    if goal:
        # Use the LLM to determine the most relevant modules for this goal
        prompt = PromptTemplate.from_template(
            """A user wants to learn about personal finance with a specific goal: {goal}
            
            Based on this goal, identify which of these financial topics are most relevant (list top 5 in order):
            
            Level 1:
            - Money Basics
            - Budgeting 101
            - Banking Essentials
            
            Level 2:
            - Emergency Fund Building
            - Debt Management
            - Credit Scores
            
            Level 3:
            - Introduction to Investing
            - Retirement Planning Basics
            - Tax Basics
            
            Level 4:
            - Advanced Investment Strategies
            - Real Estate Investing
            - Insurance Planning
            
            Level 5:
            - Estate Planning
            - Advanced Tax Strategies
            - Financial Independence Planning
            
            For each topic, explain in 1-2 sentences why it's relevant to their goal.
            Format as a JSON array with objects containing 'topic' and 'reason' keys.
            """
        )
        
        chain = prompt | llm
        response = chain.invoke({"goal": goal})
        
        # Now we need to map these topics to modules
        # This is a placeholder - in a full implementation, you would parse the JSON
        # and look up the corresponding modules
        
        # For now, let's return a default path based on user level
        return default_path(user_level)
    
    # If no specific goal, provide a default path based on user level
    return default_path(user_level)

def default_path(user_level: int) -> List[Dict[str, Any]]:
    """
    Generate a default learning path based on user level.
    
    Args:
        user_level: The user's knowledge level (1-5)
    
    Returns:
        List of recommended modules in sequence
    """
    from curriculum_roadmap import get_all_modules
    
    # Get all modules
    all_modules = get_all_modules()
    
    # Filter to modules appropriate for the user's level
    # For beginners, start at level 1
    # For others, include their current level and one level higher
    appropriate_modules = []
    
    for module in all_modules:
        module_level = int(module["level"])
        
        # For beginners (level 1), show only level 1 modules
        if user_level == 1 and module_level == 1:
            appropriate_modules.append(module)
        
        # For intermediate users, show current level and one below
        elif 1 < user_level < 5 and (module_level == user_level or module_level == user_level - 1):
            appropriate_modules.append(module)
        
        # For advanced users, show levels 4 and 5
        elif user_level >= 5 and module_level >= 4:
            appropriate_modules.append(module)
    
    # Create a path with explanations
    path = []
    for module in appropriate_modules:
        path.append({
            "module_id": module["module_id"],
            "title": module["title"],
            "description": module["description"],
            "reason": f"This module is part of {module['level_title']} and will help build your foundation."
        })
    
    return path 