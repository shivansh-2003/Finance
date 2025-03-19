---
noteId: "c041352004c611f0b66a37f0c042079e"
tags: []

---

# Personal Finance Teaching Assistant - System Architecture

This document explains the architecture of the Personal Finance Teaching Assistant, including component interactions and data flow.

## High-Level Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌────────────────┐
│                 │     │                   │     │                │
│  Streamlit UI   │◄────┤   FastAPI Server  │◄────┤  LangGraph     │
│  (streamlit_app)│     │   (main.py)       │     │  Workflow      │
│                 │─────►                   │─────►                │
└─────────────────┘     └───────────────────┘     └────────────────┘
                                 │                        │
                                 │                        │
                                 ▼                        ▼
                        ┌─────────────────┐      ┌─────────────────┐
                        │                 │      │                 │
                        │  RAG System     │◄─────┤  LLM (Claude)   │
                        │  (finance_rag)  │─────►│                 │
                        │                 │      │                 │
                        └─────────────────┘      └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐      ┌─────────────────┐
                        │                 │      │                 │
                        │  Supabase       │      │  Curriculum     │
                        │  Vector Store   │◄─────┤  Roadmap        │
                        │                 │      │                 │
                        └─────────────────┘      └─────────────────┘
```

## Component Details

### 1. User Interface (Streamlit)

The Streamlit application (`streamlit_app.py`) provides:
- Chat interface for asking financial questions
- Curriculum browser for exploring available modules
- Quiz interface for knowledge assessment
- Progress tracking dashboard
- Learning path visualization

### 2. API Layer (FastAPI)

The FastAPI server (`main.py`) provides:
- RESTful endpoints for all functionality
- Session management for conversations
- Integration with the teaching workflow
- User progress tracking

### 3. Teaching Workflow (LangGraph)

The LangGraph implementation (`langgraph-implementation.py`) manages:
- State transitions during teaching interactions
- Query analysis to determine topic type
- Context retrieval coordination
- Specialized teaching approaches for different topics

### 4. RAG System (finance_rag.py)

The Retrieval Augmented Generation system:
- Enhances queries for better retrieval
- Retrieves relevant educational content
- Formats content for teaching context
- Adapts explanations to user knowledge level

### 5. Curriculum Management (curriculum_roadmap.py)

The curriculum structure manages:
- Five progressive levels of financial education
- Module organization and prerequisites
- Topic categorization and learning paths
- Assessment structure and requirements

### 6. Database Layer (Supabase)

Supabase provides:
- Vector storage for educational content
- User progress tracking
- Session persistence
- Quiz results and assessment data

## Data Flow

### 1. User Interaction Flow

```
User Input → Streamlit UI → API Request → Query Analysis → Context Retrieval 
→ Response Generation → API Response → UI Display → User
```

### 2. Content Ingestion Flow

```
Educational Content → PDF Loader → Text Splitting → Embedding Generation 
→ Vector Storage → Supabase Database
```

### 3. Quiz Generation Flow

```
Topic Selection → User Level Determination → Quiz Generation 
→ Question Formatting → UI Presentation → Answer Collection 
→ Evaluation → Results Display → Progress Update
```

### 4. Learning Path Generation

```
User Goals/Level → Curriculum Analysis → Module Selection 
→ Path Generation → Path Visualization → UI Display
```

## Key Files and Their Purposes

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application and endpoints |
| `streamlit_app.py` | Streamlit user interface |
| `langgraph-implementation.py` | LangGraph teaching workflow |
| `finance_rag.py` | RAG implementation for finance education |
| `curriculum_roadmap.py` | Educational curriculum structure |
| `ingest_in_db.py` | Content ingestion into vector database |
| `run_finance_assistant.sh` | Helper script to run components |

## Technical Implementation Details

### State Management

The teaching workflow uses a state object with the following structure:

```python
class FinanceTeachingState(TypedDict):
    messages: List[Dict]           # Conversation history
    user_id: str                   # User identifier
    user_level: int                # Knowledge level (1-5)
    session_id: str                # Session identifier
    current_module: str            # Current module being taught
    learning_goals: List[str]      # User's learning objectives
    completed_modules: List[str]   # Modules already completed
    assessment_results: Dict       # Quiz and assessment results
    next_steps: str                # Recommended next actions
    topic_type: str                # Type of query (concept, calculation, etc.)
```

### API Endpoints

The main endpoints include:

- `/chat` - Process user messages
- `/curriculum` - Access curriculum information
- `/assessment/quiz` - Generate and evaluate quizzes
- `/learning/path` - Generate personalized learning paths
- `/user/progress` - Track user progress

### Vector Storage

Educational content is stored in Supabase with:
- 1536-dimensional embeddings (OpenAI)
- Metadata for curriculum association
- Similarity search for retrieval

## Extending the Architecture

To extend the system, consider:

1. **Adding New Agents**: Create specialized agents for specific financial topics
2. **Enhanced Visualization**: Add financial calculators and interactive visualizations
3. **User Authentication**: Implement proper authentication for multi-user support
4. **Mobile Interface**: Create a mobile-friendly version of the UI
5. **Integration with Financial Data**: Connect to financial data APIs for real-time information

## Performance Considerations

- LLM calls are the main bottleneck; implement caching for common questions
- Vector search scales well but requires proper indexing for large content bases
- Session state should be persisted in a database for production use
- Consider horizontal scaling for the API layer in multi-user scenarios 