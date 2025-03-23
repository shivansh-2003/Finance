"""
FINANCE EDUCATION RAG IMPLEMENTATION

Optimized for efficient document retrieval from Supabase vector store
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Supabase
from supabase.client import Client, create_client

# Import the curriculum content and roadmap
from curriculum_roadmap import get_module_by_id, get_next_module, get_level_assessment
from curriculum_content import get_module_content

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
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.2)

def retrieve_teaching_content(
    query: str, 
    user_id: Optional[str] = None, 
    module_id: Optional[str] = None, 
    k: int = 3
) -> str:
    """
    Retrieve educational content relevant to a user query from Supabase vector store
    and enhance it with practical examples and case studies.
    """
    # Get user level and module content
    user_level = get_user_level(user_id) if user_id else 1
    module_content = get_module_content(module_id) if module_id else {}
    
    # Create metadata filters if module ID is provided
    metadata_filters = {}
    if module_id:
        metadata_filters = {"module_id": module_id}
    
    # Enhance query for better retrieval
    enhanced_query_prompt = PromptTemplate.from_template(
        """Enhance this query to improve retrieval of personal finance educational content:
        
        Original Query: {query}
        User's Knowledge Level: {level}
        Current Module: {module_id}
        
        Add relevant financial terms and provide context to help retrieve the most appropriate content.
        Enhanced Query:"""
    )
    
    enhanced_query_chain = enhanced_query_prompt | llm | StrOutputParser()
    enhanced_query = enhanced_query_chain.invoke({
        "query": query, 
        "level": user_level,
        "module_id": module_id or "Not specified"
    })
    
    try:
        # Retrieve documents from vector store
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k, 
                "filter": metadata_filters
            }
        )
        
        docs = retriever.get_relevant_documents(enhanced_query)
        
        if not docs:
            # Fallback to module content if available
            if module_content:
                return format_module_content(module_content, query, user_level)
            return "I couldn't find specific information about this topic. Could you rephrase or be more specific?"
        
        # Combine retrieved documents with module content
        content_parts = []
        
        # Add relevant module content if available
        if module_content:
            relevant_content = extract_relevant_module_content(module_content, query)
            if relevant_content:
                content_parts.append(relevant_content)
        
        # Add retrieved documents
        for doc in docs:
            source_info = f"Source: {doc.metadata.get('title', 'Unknown Source')}"
            if "module_id" in doc.metadata:
                source_info += f" - Module: {doc.metadata.get('module_id')}"
            if "section" in doc.metadata:
                source_info += f" - Section: {doc.metadata.get('section')}"
            
            content_parts.append(f"{source_info}\n\n{doc.page_content}")
        
        # Combine and format the content
        combined_content = "\n\n---\n\n".join(content_parts)
        
        # Generate a final, coherent response
        response_prompt = ChatPromptTemplate.from_template(
            """Based on the following content, provide a comprehensive answer to the user's question.
            Make sure to include practical examples and real-world applications where relevant.
            
            User's Question: {query}
            User's Knowledge Level: {level}
            
            Content:
            {content}
            
            Provide a clear, structured response that:
            1. Directly answers the question
            2. Includes practical examples
            3. Offers actionable advice
            4. Is appropriate for the user's knowledge level
            """
        )
        
        response_chain = response_prompt | llm | StrOutputParser()
        return response_chain.invoke({
            "query": query,
            "level": user_level,
            "content": combined_content
        })
    
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return "I encountered an error while searching for information. Please try again."

def extract_relevant_module_content(module_content: dict, query: str) -> str:
    """Extract relevant content from a module based on the query."""
    relevant_parts = []
    
    # Extract key concepts
    if "key_concepts" in module_content:
        for concept, details in module_content["key_concepts"].items():
            if any(term in query.lower() for term in concept.lower().split('_')):
                relevant_parts.append(
                    f"Key Concept: {concept.replace('_', ' ').title()}\n"
                    f"Explanation: {details['explanation']}\n"
                    f"Real-world Example: {details['real_world_example']}\n"
                    f"Practice: {details['practical_exercise']}"
                )
    
    # Extract relevant case studies
    if "case_studies" in module_content:
        for case in module_content["case_studies"]:
            relevant_parts.append(
                f"Case Study: {case['title']}\n"
                f"Scenario: {case['scenario']}"
            )
    
    return "\n\n".join(relevant_parts) if relevant_parts else ""

def format_module_content(module_content: dict, query: str, user_level: int) -> str:
    """Format module content into a coherent response."""
    # Extract relevant content
    relevant_content = extract_relevant_module_content(module_content, query)
    
    if not relevant_content:
        return "I couldn't find specific information about this topic in the current module."
    
    # Generate a response using the content
    response_prompt = ChatPromptTemplate.from_template(
        """Create a comprehensive response using this educational content:
        
        {content}
        
        The response should be:
        1. Appropriate for knowledge level {level}
        2. Include practical examples and exercises
        3. Provide clear, actionable advice
        4. Connect concepts to real-world situations
        
        Response:"""
    )
    
    response_chain = response_prompt | llm | StrOutputParser()
    return response_chain.invoke({
        "content": relevant_content,
        "level": user_level
    })

def get_user_level(user_id: Optional[str]) -> int:
    """Get the user's knowledge level from their profile."""
    if not user_id:
        return 1
    
    try:
        response = supabase.table("user_progress").select("knowledge_level").eq("user_id", user_id).execute()
        if response.data and len(response.data) > 0:
            return response.data[0].get("knowledge_level", 1)
        return 1
    except Exception as e:
        print(f"Error fetching user level: {e}")
        return 1

def explain_financial_term(term: str, user_level: int = 1) -> str:
    """Get an explanation for a financial term with practical examples."""
    query = f"Definition and explanation of {term} financial term"
    
    try:
        # Check if term exists in module content
        for module_id in ["module_1_1", "module_1_2", "module_3_1"]:  # Add more as needed
            module_content = get_module_content(module_id)
            if module_content:
                for concept, details in module_content.get("key_concepts", {}).items():
                    if term.lower() in concept.lower():
                        return f"{details['explanation']}\n\nReal-world Example: {details['real_world_example']}"
        
        # Fallback to vector store
        docs = vector_store.similarity_search(query, k=2)
        
        if not docs:
            explanation_prompt = ChatPromptTemplate.from_template(
                """Explain the financial term '{term}' to someone at knowledge level {level}.
                
                Provide a clear, concise explanation that:
                1. Starts with a simple definition
                2. Explains why this term is important
                3. Gives a real-world example
                4. Is appropriate for a level {level} learner
                """
            )
            
            chain = explanation_prompt | llm | StrOutputParser()
            return chain.invoke({"term": term, "level": user_level})
        
        # Process retrieved documents
        content_parts = [doc.page_content for doc in docs]
        retrieved_content = "\n\n".join(content_parts)
        
        # Format the explanation
        explanation_prompt = ChatPromptTemplate.from_template(
            """Explain the financial term '{term}' based on this retrieved information:
            
            {content}
            
            Create an explanation that:
            1. Is clear and understandable
            2. Matches the knowledge level of {level}
            3. Includes key points from the retrieved information
            4. Provides practical context
            """
        )
        
        chain = explanation_prompt | llm | StrOutputParser()
        return chain.invoke({
            "term": term, 
            "content": retrieved_content, 
            "level": user_level
        })
    
    except Exception as e:
        print(f"Error explaining financial term: {e}")
        return f"I couldn't find a detailed explanation for '{term}'. Please try another term."

def get_practical_examples(concept: str, difficulty: str = "beginner") -> str:
    """Get practical examples for a financial concept."""
    # First, check module content for examples
    for module_id in ["module_1_1", "module_1_2", "module_3_1"]:  # Add more as needed
        module_content = get_module_content(module_id)
        if module_content:
            for concept_name, details in module_content.get("key_concepts", {}).items():
                if concept.lower() in concept_name.lower():
                    return f"{details['real_world_example']}\n\nPractical Exercise: {details['practical_exercise']}"
    
    # Fallback to vector store
    query = f"Practical examples of {concept} in {difficulty} personal finance"
    
    try:
        docs = vector_store.similarity_search(query, k=2)
        
        if not docs:
            examples_prompt = ChatPromptTemplate.from_template(
                """Generate practical examples for the financial concept '{concept}' 
                at a {difficulty} level.
                
                Create 2-3 clear, realistic scenarios that:
                1. Illustrate the concept in action
                2. Are appropriate for a {difficulty} level understanding
                3. Include specific, relatable details
                4. Show the practical implications of the concept
                """
            )
            
            chain = examples_prompt | llm | StrOutputParser()
            return chain.invoke({"concept": concept, "difficulty": difficulty})
        
        content_parts = [doc.page_content for doc in docs]
        retrieved_content = "\n\n".join(content_parts)
        
        examples_prompt = ChatPromptTemplate.from_template(
            """Create practical examples for '{concept}' based on this retrieved information:
            
            {content}
            
            Generate 2-3 examples that:
            1. Are clear and illustrative
            2. Match the {difficulty} level of understanding
            3. Incorporate key points from the retrieved content
            4. Provide actionable insights
            """
        )
        
        chain = examples_prompt | llm | StrOutputParser()
        return chain.invoke({
            "concept": concept, 
            "content": retrieved_content, 
            "difficulty": difficulty
        })
    
    except Exception as e:
        print(f"Error retrieving practical examples: {e}")
        return f"I couldn't find specific examples for '{concept}'. Let me provide a general explanation."

def recommend_next_topic(user_id: str) -> Dict[str, Any]:
    """Recommend what to study next based on user's progress and interests."""
    user_progress = get_user_progress(user_id)
    user_level = user_progress.get("knowledge_level", 1)
    completed_modules = user_progress.get("completed_modules", [])
    
    # Get next module in sequence if user has completed modules
    if completed_modules:
        last_completed = completed_modules[-1]
        next_module = get_next_module(last_completed)
        if next_module:
            module_content = get_module_content(next_module["module_id"])
            return {
                **next_module,
                "content_preview": module_content,
                "reason": "This module builds on your previous learning and introduces new concepts."
            }
    
    # For new users or if no clear next module
    starter_module = {
        "module_id": "module_1_1",
        "title": "Money Basics",
        "description": "Understanding the fundamental concepts of money and personal finance",
        "content_preview": get_module_content("module_1_1"),
        "reason": "This module provides essential foundations for personal finance understanding."
    }
    
    return starter_module

def get_user_progress(user_id: str) -> Dict[str, Any]:
    """Get detailed user progress information."""
    try:
        response = supabase.table("user_progress").select("*").eq("user_id", user_id).execute()
        
        if response.data and len(response.data) > 0:
            progress_data = response.data[0]
            
            # Enhance progress data with module details
            completed_modules = progress_data.get("completed_modules", [])
            enhanced_modules = []
            
            for module_id in completed_modules:
                module = get_module_by_id(module_id)
                if module:
                    module_content = get_module_content(module_id)
                    enhanced_modules.append({
                        **module,
                        "completion_date": progress_data.get("completion_dates", {}).get(module_id),
                        "key_concepts": list(module_content.get("key_concepts", {}).keys())
                    })
            
            return {
                **progress_data,
                "enhanced_modules": enhanced_modules,
                "next_recommended": recommend_next_topic(user_id)
            }
        
        return {
            "completed_modules": [],
            "last_completed_module": None,
            "knowledge_level": 1,
            "quiz_scores": {},
            "enhanced_modules": [],
            "next_recommended": recommend_next_topic(user_id)
        }
        
    except Exception as e:
        print(f"Error fetching user progress: {e}")
        return {
            "completed_modules": [],
            "last_completed_module": None,
            "knowledge_level": 1,
            "quiz_scores": {},
            "enhanced_modules": [],
            "next_recommended": recommend_next_topic(user_id)
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