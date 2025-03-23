# Personal Finance Teaching Assistant

An intelligent personal finance teaching assistant built to simplify learning about money management.

## Overview

This project uses Supabase for database management, FastAPI for API endpoints, and Streamlit for a user-friendly interface. It delivers a structured curriculum covering personal finance topics from basics to advanced strategies, while tracking user progress and providing personalized recommendations.

## Features

- **Structured Curriculum:** Step-by-step modules for learning personal finance.
- **User Progress Tracking:** Monitor your progress and get recommendations based on your performance.
- **Interactive Chat & Quizzes:** Ask questions and test your understanding with quizzes.
- **Simplified Setup:** Ready-to-use Supabase tables for documents, feedback, user_progress, and sessions; combined with ingestion of financial documents into the vector store.

## Setup & Installation

1. **Clone the Repository & Install Dependencies:**
   ```bash
   git clone <repository_url>
   cd finance-teaching-assistant
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Setup Environment Variables:**
   Create a `.env` file (or copy from `.env.example`) and update with your credentials:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

3. **Database Setup:**
   Ensure your Supabase project has the following tables: `documents` (with vector embeddings), `feedback`, `user_progress`, and `sessions`.

4. **Ingest Educational Content:**
   Run the ingestion script to load your financial documents into the Supabase vector store:
   ```bash
   python Finace_edu/ingest_in_db.py
   ```

5. **Run the Applications:**
   - Start the API server:
     ```bash
     python Finace_edu/main.py
     ```
   - Launch the Streamlit UI:
     ```bash
     streamlit run Finace_edu/streamlit_app.py
     ```

## API Endpoints

- **POST /chat** - Process a user question and return a personalized finance explanation.
- **GET /curriculum** - Retrieve the full curriculum roadmap.
- **GET /curriculum/module/{module_id}** - Get details for a specific module.
- **POST /learning/complete-module** - Mark a module as completed.
- **POST /assessment/quiz** - Generate a quiz on a finance topic.
- **POST /assessment/submit-quiz** - Submit quiz answers and receive feedback.
- **POST /learning/path** - Get a personalized learning path based on your goals.
- **GET /user/progress/{user_id}** - Check your progress and performance.

## Usage

- **Chat:** Ask questions about personal finance and get dynamic, tailored responses.
- **Browse the Curriculum:** Explore the financial lessons via the Streamlit UI.
- **Quizzes:** Test your understanding with interactive assessments.
- **Progress Tracking:** Monitor your learning journey and receive next-step recommendations.

## License

This project is licensed under the MIT License. 