# Personal Finance Education Platform - Core Implementation
# This file demonstrates the key components of the system

import os
from typing import List, Dict, Any
from langchain.chat_models import ChatAnthropic
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from langchain.tools import TavilySearchResults
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Environment setup (would use dotenv in production)
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # For embeddings
os.environ["SUPABASE_URL"] = "your-supabase-url"
os.environ["SUPABASE_SERVICE_KEY"] = "your-supabase-key"
os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"

# Initialize LLM
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.2)

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Establish connection to Supabase Vector DB
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client_url=os.environ.get("SUPABASE_URL"),
    client_key=os.environ.get("SUPABASE_SERVICE_KEY"),
    table_name="documents"
)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Initialize Tavily search
search_tool = TavilySearchResults(api_key=os.environ.get("TAVILY_API_KEY"))

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Curriculum data structure
class CurriculumTopic:
    def __init__(self, level, week, title, subtopics):
        self.level = level
        self.week = week
        self.title = title
        self.subtopics = subtopics

# Parse curriculum from markdown (simplified)
def parse_curriculum(curriculum_md):
    """Parse curriculum markdown into structured format"""
    # This would parse the actual markdown in production
    # Simplified example structure
    curriculum = [
        CurriculumTopic(
            level=1, 
            week=1, 
            title="Money Basics", 
            subtopics=[
                "Understanding income and expenses",
                "Creating a personal balance sheet",
                "Setting financial goals",
                "Financial mindset fundamentals"
            ]
        ),
        # Additional topics would be parsed from markdown
    ]
    return curriculum

# Load curriculum
curriculum = parse_curriculum("curriculum.md")

# Knowledge Retrieval System
class KnowledgeRetrievalSystem:
    def __init__(self, qa_chain, search_tool, llm):
        self.qa_chain = qa_chain
        self.search_tool = search_tool
        self.llm = llm
        self._cache = {}  # Simple in-memory cache
    
    def _get_cache_key(self, query: str, use_internet: bool) -> str:
        """Generate a cache key from query parameters"""
        return f"{query}_{use_internet}"
    
    def retrieve_information(self, query: str, use_internet: bool = True):
        """Retrieve information using vector DB and optionally internet search with caching"""
        cache_key = self._get_cache_key(query, use_internet)
        
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Query vector database
        vector_results = self.qa_chain.invoke({"query": query})
        source_docs = vector_results.get("source_documents", [])
        has_sufficient_info = len(source_docs) >= 3
        
        if has_sufficient_info or not use_internet:
            result = (vector_results["result"], source_docs)
            self._cache[cache_key] = result
            return result
        
        # Augment with internet search
        search_results = self.search_tool.invoke({"query": query})
        search_context = "\n\n".join([
            f"Title: {r['title']}\nContent: {r['content']}" 
            for r in search_results[:3]
        ])
        
        # Create combined prompt
        combined_prompt = f"""
        Create a comprehensive response about "{query}" using both sets of information below.
        
        Vector database information:
        {vector_results["result"]}
        
        Internet search information:
        {search_context}
        
        Synthesize this information into a well-structured educational response about "{query}".
        Focus on accuracy, clarity, and educational value.
        """
        
        # Generate enhanced response
        enhanced_response = self.llm.invoke(combined_prompt)
        
        # Combine sources
        all_sources = source_docs + [
            Document(page_content=r["content"], metadata={"source": r["url"], "title": r["title"]})
            for r in search_results[:3]
        ]
        
        result = (enhanced_response.content, all_sources)
        self._cache[cache_key] = result
        return result

# Initialize Knowledge Retrieval System
knowledge_system = KnowledgeRetrievalSystem(qa_chain, search_tool, llm)

# Curriculum Agent
class CurriculumAgent:
    def __init__(self, curriculum, knowledge_system, llm):
        self.curriculum = curriculum
        self.knowledge_system = knowledge_system
        self.llm = llm
        self.user_progress = {}  # Track user progress
    
    def get_topic_content(self, level, week, subtopic=None, user_id=None):
        """Get content for a specific topic or subtopic with personalization"""
        # Find the requested topic
        topic = next((t for t in self.curriculum 
                     if t.level == level and t.week == week), None)
        
        if not topic:
            return "Topic not found in curriculum"
        
        # Get user's progress if available
        user_progress = self.user_progress.get(user_id, {}) if user_id else {}
        completed_topics = user_progress.get('completed_topics', set())
        
        # Construct query based on whether subtopic is specified
        if subtopic:
            query = f"Explain {subtopic} in the context of {topic.title} for personal finance education"
        else:
            query = f"Provide an overview of {topic.title} for personal finance education"
        
        # Add personalization based on progress
        if user_id and completed_topics:
            related_completed = [t for t in completed_topics 
                               if abs(t[0] - level) <= 1 and abs(t[1] - week) <= 2]
            if related_completed:
                query += f"\nConnect this to previously learned topics from weeks {', '.join(str(w) for l,w,_ in related_completed)}"
        
        # Retrieve information
        content, sources = self.knowledge_system.retrieve_information(query)
        
        # Format response with sources and next steps
        formatted_response = f"""
        # {topic.title}
        {'' if not subtopic else f'## {subtopic}'}
        
        {content}
        
        ## Recommended Next Steps
        {self._get_next_steps(level, week, subtopic, user_id)}
        
        ## Sources
        {self._format_sources(sources)}
        """
        
        return formatted_response
    
    def _get_next_steps(self, level, week, subtopic, user_id=None):
        """Generate personalized next steps recommendations"""
        if not user_id:
            return self._get_default_next_steps(level, week, subtopic)
        
        user_progress = self.user_progress.get(user_id, {})
        completed_topics = user_progress.get('completed_topics', set())
        
        # Find gaps in knowledge
        current_level_topics = [t for t in self.curriculum if t.level == level]
        incomplete_topics = []
        for topic in current_level_topics:
            if (level, topic.week, None) not in completed_topics:
                incomplete_topics.append(topic)
        
        # Generate recommendations
        recommendations = []
        if incomplete_topics:
            next_topic = min(incomplete_topics, key=lambda t: t.week)
            recommendations.append(f"📚 Next Topic: Week {next_topic.week} - {next_topic.title}")
        
        # Add skill-building recommendations
        if level < 5:  # Not at max level
            next_level = next((t for t in self.curriculum if t.level == level + 1), None)
            if next_level:
                recommendations.append(f"🎯 Next Level Preview: {next_level.title}")
        
        # Add practice recommendations
        recommendations.append("💪 Practice Exercises:")
        recommendations.extend([
            "- Complete the topic quiz",
            "- Try the hands-on exercises",
            "- Review related case studies"
        ])
        
        return "\n".join(recommendations)
    
    def _get_default_next_steps(self, level, week, subtopic):
        """Get default next steps without personalization"""
        next_topic = next((t for t in self.curriculum 
                          if t.level == level and t.week > week), None)
        
        if not next_topic:
            next_topic = next((t for t in self.curriculum 
                             if t.level == level + 1), None)
        
        recommendations = [
            f"📚 Next Topic: {next_topic.title if next_topic else 'Complete current level'}"
        ]
        
        recommendations.extend([
            "💪 Recommended Actions:",
            "- Take the topic assessment",
            "- Practice with real-world examples",
            "- Review key concepts"
        ])
        
        return "\n".join(recommendations)
    
    def update_progress(self, user_id, level, week, subtopic=None, completed=True):
        """Update user's progress in the curriculum"""
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {'completed_topics': set()}
        
        if completed:
            self.user_progress[user_id]['completed_topics'].add((level, week, subtopic))
        else:
            self.user_progress[user_id]['completed_topics'].discard((level, week, subtopic))
    
    def get_progress_summary(self, user_id):
        """Get a summary of user's progress"""
        if user_id not in self.user_progress:
            return "No progress recorded"
        
        completed = self.user_progress[user_id]['completed_topics']
        total_topics = sum(1 for t in self.curriculum)
        completed_count = len(completed)
        
        return {
            'completed_count': completed_count,
            'total_topics': total_topics,
            'progress_percentage': (completed_count / total_topics) * 100,
            'completed_topics': sorted(list(completed))
        }
    
    def _format_sources(self, sources):
        """Format sources for display"""
        formatted = []
        for i, doc in enumerate(sources, 1):
            source = doc.metadata.get("source", "Unknown source")
            title = doc.metadata.get("title", "Untitled")
            formatted.append(f"{i}. {title} ({source})")
        
        return "\n".join(formatted)

# Assessment Agent
class AssessmentAgent:
    def __init__(self, knowledge_system, llm):
        self.knowledge_system = knowledge_system
        self.llm = llm
        self.assessment_history = {}  # Track assessment history
        self.difficulty_levels = {
            1: "beginner",
            2: "beginner",
            3: "intermediate",
            4: "intermediate",
            5: "advanced"
        }
    
    def generate_quiz(self, topic, subtopic=None, user_level=1, previous_scores=None):
        """Generate an adaptive quiz based on user's level and performance"""
        # Determine difficulty
        difficulty = self._determine_difficulty(user_level, previous_scores)
        
        # Construct query
        query = f"{topic}"
        if subtopic:
            query = f"{subtopic} in {topic}"
        
        # Get content about the topic
        content, _ = self.knowledge_system.retrieve_information(
            f"Explain {query} for personal finance education"
        )
        
        # Create quiz generation prompt
        quiz_prompt = f"""
        Create a {difficulty} level quiz about {query}.
        
        Content to base questions on:
        {content}
        
        Create 5 multiple-choice questions that:
        1. Test understanding at a {difficulty} level
        2. Include practical applications
        3. Build from simpler to more complex concepts
        4. Cover different aspects of the topic
        5. Include at least one calculation-based question
        
        For each question:
        - Make it challenging but fair for {difficulty} level
        - Ensure distractors are plausible but clearly incorrect
        - Provide a detailed explanation of why the correct answer is right
        - Include a "Key Learning Point" after the explanation
        
        Format each question as:
        
        Question: [Question text]
        A. [Option A]
        B. [Option B]
        C. [Option C]
        D. [Option D]
        Correct Answer: [Letter]
        Explanation: [Detailed explanation]
        Key Learning Point: [Concise learning takeaway]
        """
        
        # Generate quiz
        quiz_response = self.llm.invoke(quiz_prompt)
        return quiz_response.content
    
    def _determine_difficulty(self, user_level, previous_scores=None):
        """Determine appropriate difficulty based on user level and performance"""
        base_difficulty = self.difficulty_levels[user_level]
        
        if not previous_scores:
            return base_difficulty
        
        # Calculate average of last 3 scores
        recent_scores = previous_scores[-3:]
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # Adjust difficulty based on performance
        if base_difficulty == "beginner":
            if avg_score > 0.85:
                return "intermediate"
        elif base_difficulty == "intermediate":
            if avg_score > 0.85:
                return "advanced"
            elif avg_score < 0.60:
                return "beginner"
        elif base_difficulty == "advanced":
            if avg_score < 0.60:
                return "intermediate"
        
        return base_difficulty
    
    def generate_custom_assessment(self, custom_topic, user_level=1):
        """Generate a comprehensive custom assessment"""
        # Get information about the custom topic
        content, _ = self.knowledge_system.retrieve_information(
            f"Explain {custom_topic} in the context of personal finance",
            use_internet=True
        )
        
        difficulty = self.difficulty_levels[user_level]
        
        # Create assessment prompt
        assessment_prompt = f"""
        Create a comprehensive {difficulty}-level assessment about {custom_topic}.
        
        Content to base assessment on:
        {content}
        
        Include the following sections:

        1. Knowledge Check (5 multiple-choice questions)
        - Progress from basic to advanced understanding
        - Include practical scenarios
        - Provide detailed explanations for all answers
        
        2. Case Study Analysis
        - Create a realistic financial scenario
        - Include relevant numbers and data
        - Ask 2-3 analysis questions that require critical thinking
        - Provide a rubric for evaluating responses
        
        3. Practical Application
        - Give a hands-on exercise related to {custom_topic}
        - Include step-by-step instructions
        - Provide example calculations if relevant
        - List expected learning outcomes
        
        4. Self-Reflection Questions
        - Ask 2 questions about applying this knowledge
        - Include prompts for personal financial planning
        - Encourage critical thinking about financial decisions
        
        5. Additional Resources
        - Suggest 2-3 reliable sources for further learning
        - Include tools or calculators if relevant
        
        Format the assessment professionally with clear sections and instructions.
        Make it challenging but appropriate for {difficulty} level.
        """
        
        # Generate assessment
        assessment_response = self.llm.invoke(assessment_prompt)
        
        return assessment_response.content
    
    def evaluate_response(self, question, user_answer, correct_answer, difficulty):
        """Provide detailed feedback on user's response"""
        evaluation_prompt = f"""
        Analyze this response to a {difficulty}-level financial question:
        
        Question: {question}
        User's Answer: {user_answer}
        Correct Answer: {correct_answer}
        
        Provide feedback that:
        1. Identifies what the user understood correctly
        2. Points out any misconceptions
        3. Explains the correct reasoning
        4. Gives a specific tip for improvement
        5. Connects to practical financial applications
        
        Keep the tone encouraging while being clear about areas for improvement.
        """
        
        feedback = self.llm.invoke(evaluation_prompt)
        return feedback.content
    
    def track_performance(self, user_id, topic, score, difficulty):
        """Track user's assessment performance"""
        if user_id not in self.assessment_history:
            self.assessment_history[user_id] = []
        
        self.assessment_history[user_id].append({
            'topic': topic,
            'score': score,
            'difficulty': difficulty,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_performance_summary(self, user_id):
        """Get summary of user's assessment performance"""
        if user_id not in self.assessment_history:
            return "No assessment history available"
        
        history = self.assessment_history[user_id]
        total_assessments = len(history)
        avg_score = sum(entry['score'] for entry in history) / total_assessments
        
        # Calculate improvement trend
        if total_assessments >= 2:
            recent_scores = [entry['score'] for entry in sorted(history, key=lambda x: x['timestamp'])[-5:]]
            trend = sum(b - a for a, b in zip(recent_scores[:-1], recent_scores[1:])) / (len(recent_scores) - 1)
        else:
            trend = 0
        
        return {
            'total_assessments': total_assessments,
            'average_score': avg_score,
            'improvement_trend': trend,
            'recent_topics': [entry['topic'] for entry in sorted(history, key=lambda x: x['timestamp'])[-5:]],
            'current_level': self._determine_current_level(avg_score, total_assessments)
        }
    
    def _determine_current_level(self, avg_score, total_assessments):
        """Determine user's current level based on performance"""
        if total_assessments < 3:
            return 1
        
        if avg_score >= 0.85:
            return min(5, total_assessments // 5 + 2)
        elif avg_score >= 0.70:
            return min(4, total_assessments // 6 + 1)
        else:
            return max(1, total_assessments // 8)

# Initialize agents
curriculum_agent = CurriculumAgent(curriculum, knowledge_system, llm)
assessment_agent = AssessmentAgent(knowledge_system, llm)

# Example workflow functions

def get_topic_content(level, week, subtopic=None):
    """Get content for a topic or subtopic"""
    return curriculum_agent.get_topic_content(level, week, subtopic)

def generate_topic_quiz(level, week, subtopic=None):
    """Generate a quiz for a topic"""
    topic = next((t for t in curriculum if t.level == level and t.week == week), None)
    if not topic:
        return "Topic not found"
    
    return assessment_agent.generate_quiz(topic.title, subtopic)

def generate_custom_assessment(topic):
    """Generate a custom assessment on a user-chosen topic"""
    return assessment_agent.generate_custom_assessment(topic)

# API routes would be implemented here 
# (using FastAPI, Flask, etc.)
"""
@app.get("/curriculum")
def get_curriculum():
    return curriculum

@app.get("/topic/{level}/{week}")
def get_topic(level: int, week: int, subtopic: str = None):
    return get_topic_content(level, week, subtopic)

@app.get("/assessment/{level}/{week}")
def get_assessment(level: int, week: int, subtopic: str = None):
    return generate_topic_quiz(level, week, subtopic)

@app.post("/custom-assessment")
def custom_assessment(topic: str):
    return generate_custom_assessment(topic)
"""

# Example of LangGraph implementation (simplified)
def create_workflow():
    """Create an optimized workflow for the finance education system"""
    # Define tool for curriculum navigation
    curriculum_tools = [
        Tool(
            name="retrieve_topic_content",
            func=get_topic_content,
            description="Get educational content for a specific topic in the personal finance curriculum"
        ),
        Tool(
            name="get_progress_summary",
            func=curriculum_agent.get_progress_summary,
            description="Get a summary of user's progress in the curriculum"
        ),
        Tool(
            name="recommend_next_steps",
            func=lambda level, week, subtopic, user_id: curriculum_agent._get_next_steps(level, week, subtopic, user_id),
            description="Get personalized recommendations for next steps in learning"
        )
    ]
    
    # Define tool for assessment generation
    assessment_tools = [
        Tool(
            name="generate_quiz",
            func=generate_topic_quiz,
            description="Generate a quiz for a specific topic in the personal finance curriculum"
        ),
        Tool(
            name="generate_custom_assessment",
            func=generate_custom_assessment,
            description="Generate a custom assessment on a user-chosen personal finance topic"
        ),
        Tool(
            name="evaluate_response",
            func=assessment_agent.evaluate_response,
            description="Evaluate and provide feedback on user's assessment responses"
        ),
        Tool(
            name="get_performance_summary",
            func=assessment_agent.get_performance_summary,
            description="Get a summary of user's assessment performance"
        )
    ]
    
    # Create agent prompt templates with improved context
    curriculum_prompt = ChatPromptTemplate.from_template(
        """You are a personal finance education agent that helps users navigate a curriculum.
        You have access to the user's progress and can provide personalized recommendations.
        
        Current user level: {user_level}
        Progress summary: {progress_summary}
        
        Based on the user's request and progress, provide appropriate educational content
        and recommendations for their learning journey.
        
        User request: {input}
        
        Remember to:
        1. Consider their current level and progress
        2. Connect new topics to previously learned material
        3. Provide clear next steps
        4. Encourage consistent progress
        
        {agent_scratchpad}
        """
    )
    
    assessment_prompt = ChatPromptTemplate.from_template(
        """You are an assessment agent for personal finance education.
        You can generate adaptive assessments based on user's level and performance.
        
        Current user level: {user_level}
        Performance summary: {performance_summary}
        
        Based on the user's request and performance history, generate appropriate
        assessments and provide detailed feedback.
        
        User request: {input}
        
        Remember to:
        1. Adjust difficulty based on performance
        2. Provide detailed explanations
        3. Give constructive feedback
        4. Track progress over time
        
        {agent_scratchpad}
        """
    )
    
    # Create agents with error handling
    try:
        curriculum_agent = create_structured_chat_agent(
            llm=llm,
            tools=curriculum_tools,
            prompt=curriculum_prompt
        )
        
        assessment_agent = create_structured_chat_agent(
            llm=llm,
            tools=assessment_tools,
            prompt=assessment_prompt
        )
    except Exception as e:
        raise Exception(f"Failed to create agents: {str(e)}")
    
    # Create state graph
    workflow = StateGraph()
    
    # Add nodes with error handling
    try:
        workflow.add_node("curriculum_agent", AgentExecutor(agent=curriculum_agent))
        workflow.add_node("assessment_agent", AgentExecutor(agent=assessment_agent))
        
        # Add error handling node
        workflow.add_node("error_handler", _handle_errors)
    except Exception as e:
        raise Exception(f"Failed to add nodes to workflow: {str(e)}")
    
    # Define transitions with conditional routing
    try:
        # Add edges with conditions
        workflow.add_conditional_edges(
            "curriculum_agent",
            {
                "assessment_agent": lambda x: "assessment" in x.lower() or "quiz" in x.lower(),
                "error_handler": lambda x: "error" in x.lower(),
                "curriculum_agent": lambda x: True  # Default path
            }
        )
        
        workflow.add_conditional_edges(
            "assessment_agent",
            {
                "curriculum_agent": lambda x: "learn" in x.lower() or "topic" in x.lower(),
                "error_handler": lambda x: "error" in x.lower(),
                "assessment_agent": lambda x: True  # Default path
            }
        )
        
        # Error handler always returns to appropriate agent
        workflow.add_conditional_edges(
            "error_handler",
            {
                "curriculum_agent": lambda x: "curriculum" in x.lower(),
                "assessment_agent": lambda x: "assessment" in x.lower()
            }
        )
        
        workflow.set_entry_point("curriculum_agent")
    except Exception as e:
        raise Exception(f"Failed to add edges to workflow: {str(e)}")
    
    # Compile graph with validation
    try:
        compiled_workflow = workflow.compile()
        return compiled_workflow
    except Exception as e:
        raise Exception(f"Failed to compile workflow: {str(e)}")

def _handle_errors(error_state):
    """Handle errors in the workflow gracefully"""
    error_message = error_state.get("error", "An unknown error occurred")
    
    # Log error for monitoring (would use proper logging in production)
    print(f"Workflow error: {error_message}")
    
    # Create user-friendly error message
    user_message = f"""I apologize, but I encountered an issue while processing your request.
    
    To help you continue learning:
    1. Try refreshing the page
    2. Rephrase your question if it was unclear
    3. Break down complex requests into smaller steps
    
    If the issue persists, please contact support.
    
    Technical details: {error_message}
    """
    
    return {"message": user_message, "success": False}

# For testing purposes
if __name__ == "__main__":
    # Example usage
    topic_content = get_topic_content(1, 1, "Understanding income and expenses")
    print("\n\nTOPIC CONTENT EXAMPLE:")
    print(topic_content)
    
    quiz = generate_topic_quiz(1, 1, "Understanding income and expenses")
    print("\n\nQUIZ EXAMPLE:")
    print(quiz)
    
    custom = generate_custom_assessment("Debt snowball vs. avalanche method")
    print("\n\nCUSTOM ASSESSMENT EXAMPLE:")
    print(custom)
