---
noteId: "d3274fd004c611f0b66a37f0c042079e"
tags: []

---

# Personal Finance Teaching Assistant

An intelligent educational agent that teaches users about personal finance concepts from basics to advanced topics, using a structured curriculum and personalized learning approach.

## Overview

This project implements a comprehensive teaching assistant for personal finance education using:
- Claude 3.7 Sonnet as the Large Language Model
- LangChain for creating specialized teaching chains
- LangGraph for orchestrating the teaching workflow
- Supabase for user progress tracking and vector storage
- FastAPI for creating the API endpoints

The assistant is designed to provide personalized finance education through:
1. A structured curriculum from beginner to advanced levels
2. Personalized learning paths based on user goals and current knowledge
3. Adaptive teaching approaches for different types of financial topics
4. Interactive quizzes and assessments
5. Progress tracking and recommendations

## Project Structure

```
Finace_edu/
├── curriculum_roadmap.py  # Defines the complete educational curriculum
├── finance_rag.py         # RAG implementation for finance education
├── langgraph-implementation.py  # State management workflow using LangGraph
├── main.py                # FastAPI application and endpoints
├── ingest_in_db.py        # Script to ingest educational content
├── .env                   # Environment variables (not in repo)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Curriculum Structure

The curriculum is organized into 5 progressive levels:

1. **Personal Finance Foundations** - For absolute beginners
   - Money Basics
   - Budgeting 101
   - Banking Essentials

2. **Building Financial Security** - Establishing good financial habits
   - Emergency Fund Building
   - Debt Management
   - Credit Scores

3. **Growing Your Wealth** - Introduction to investing
   - Introduction to Investing
   - Retirement Planning Basics
   - Tax Basics

4. **Advanced Financial Strategies** - Optimizing finances
   - Advanced Investment Strategies
   - Real Estate Investing
   - Insurance Planning

5. **Financial Mastery** - Complex financial concepts
   - Estate Planning
   - Advanced Tax Strategies
   - Financial Independence Planning

Each module includes topics, exercises, and assessments appropriate for the user's level.

## Setup and Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/finance-teaching-assistant.git
   cd finance-teaching-assistant
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

5. Setup your Supabase database
   - Create the tables as defined in implementation-plan.py
   - Enable vector storage functionality

6. Prepare and ingest educational content
   ```bash
   python Finace_edu/ingest_in_db.py
   ```

7. Start the API server
   ```bash
   python Finace_edu/main.py
   ```

## API Endpoints

The API provides the following endpoints:

- **POST /chat** - Process a user message and get an educational response
- **GET /curriculum** - Get the full curriculum roadmap
- **GET /curriculum/modules/level/{level}** - Get modules for a specific level
- **GET /curriculum/module/{module_id}** - Get details for a specific module
- **POST /learning/complete-module** - Mark a module as completed for a user
- **POST /assessment/quiz** - Generate a quiz on a specific topic
- **POST /assessment/submit-quiz** - Submit quiz answers and get results
- **POST /learning/path** - Get a personalized learning path
- **GET /user/progress/{user_id}** - Get a user's learning progress

See the FastAPI Swagger documentation at `http://localhost:8000/docs` for detailed API specifications.

## Key Components

### Curriculum Roadmap

The curriculum is structured as a progressive learning journey from basic to advanced financial concepts, with each module building on previous knowledge.

### LangGraph Workflow

The teaching workflow uses LangGraph for state management, allowing the system to:
1. Analyze user queries to determine the topic type
2. Retrieve relevant educational content
3. Choose the appropriate teaching approach
4. Generate personalized explanations
5. Recommend next steps in the learning journey

### RAG Implementation

The RAG (Retrieval Augmented Generation) system combines:
- Vector search for retrieving relevant financial educational content
- Query enhancement for better retrieval accuracy
- Context-aware response generation tailored to the user's knowledge level

### Teaching Approaches

The system adapts its teaching approach based on what the user is asking about:
- **Concept explanations** - For understanding financial terms and concepts
- **Calculation guidance** - For financial formulas and calculations
- **Strategy teaching** - For approaches to financial decisions
- **Tool education** - For understanding financial products and services
- **Knowledge assessment** - For testing comprehension with quizzes

## Extending the System

### Adding New Content

1. Create educational content in markdown or PDF format
2. Organize files following the curriculum structure
3. Add metadata about difficulty level and topics
4. Run the ingestion script to add content to the vector database

### Customizing the Curriculum

Modify `curriculum_roadmap.py` to:
- Add new modules or topics
- Change the sequence of learning
- Adjust difficulty levels
- Create specialized learning paths

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Built with Claude 3.7 Sonnet by Anthropic
- Uses LangChain and LangGraph frameworks
- Utilizes Supabase for database functionality 