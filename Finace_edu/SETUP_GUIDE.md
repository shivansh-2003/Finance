---
noteId: "c1e2577004c511f0b66a37f0c042079e"
tags: []

---

# Personal Finance Teaching Assistant - Setup Guide

This guide provides step-by-step instructions to set up and run the Personal Finance Teaching Assistant project on your local machine.

## Prerequisites

- Python 3.9+ installed on your system
- Git (optional, for cloning the repository)
- An OpenAI API key (for embeddings and LLM access)
- An Anthropic API key (for Claude 3.7 Sonnet)
- A Supabase account and project setup

## Step 1: Clone or Download the Repository

```bash
git clone https://github.com/yourusername/finance-teaching-assistant.git
cd finance-teaching-assistant
```

Or download and extract the ZIP file of the project.

## Step 2: Set Up the Environment

### Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Set Up Supabase

1. Create a new Supabase project at https://supabase.com
2. After project creation, go to Settings → API to get your:
   - Project URL (SUPABASE_URL)
   - API Key (SUPABASE_KEY) - use the "anon" public key

3. Set up the necessary tables in Supabase:

   a. Go to the SQL Editor in your Supabase dashboard
   
   b. Run the following SQL to create the required tables:

   ```sql
   -- Vector store table for educational content
   CREATE TABLE documents (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     content TEXT,
     metadata JSONB,
     embedding VECTOR(1536)
   );

   -- User progress tracking
   CREATE TABLE user_progress (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     user_id TEXT,
     knowledge_level INTEGER DEFAULT 1,
     completed_modules TEXT[] DEFAULT '{}',
     last_completed_module TEXT,
     quiz_scores JSONB DEFAULT '{}',
     last_interaction TIMESTAMPTZ DEFAULT NOW()
   );

   -- Create the vector search function
   CREATE OR REPLACE FUNCTION match_documents(
     query_embedding VECTOR(1536),
     match_threshold FLOAT DEFAULT 0.5,
     match_count INT DEFAULT 5
   ) RETURNS TABLE (
     id UUID,
     content TEXT,
     metadata JSONB,
     similarity FLOAT
   ) LANGUAGE plpgsql AS $$
   BEGIN
     RETURN QUERY
     SELECT
       documents.id,
       documents.content,
       documents.metadata,
       1 - (documents.embedding <=> query_embedding) AS similarity
     FROM documents
     WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
     ORDER BY similarity DESC
     LIMIT match_count;
   END;
   $$;
   ```

## Step 4: Configure Environment Variables

1. Create a `.env` file in the project root by copying the example:

```bash
cp .env.example .env
```

2. Open the `.env` file and add your API keys and Supabase credentials:

```
# API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Application Settings
APP_PORT=8000
APP_HOST=0.0.0.0
DEBUG_MODE=True

# Vector Store Settings
VECTOR_DIMENSIONS=1536
```

## Step 5: Prepare Educational Content

1. Create a `documents` directory in the project root:

```bash
mkdir -p documents
```

2. Add your educational content as PDF files to this directory. For example:
   - `intro_to_finance.pdf`
   - `budgeting_basics.pdf`
   - `investing_101.pdf`

If you don't have content ready, you can create simple text files with personal finance information.

## Step 6: Ingest Content into Supabase

Run the ingestion script to embed and store your educational content:

```bash
cd Finace_edu
python ingest_in_db.py
```

If successful, you should see output indicating documents were processed and stored in Supabase.

## Step 7: Run the API Server

Start the FastAPI server:

```bash
cd Finace_edu
python main.py
```

The server should start and display a message like:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Step 8: Run the Streamlit Interface

Open a new terminal window, activate the virtual environment again, and run:

```bash
cd Finace_edu
streamlit run streamlit_app.py
```

This should automatically open your browser with the Streamlit interface. If not, you can access it at `http://localhost:8501`.

## Step 9: Using the Application

1. **API Documentation**:
   - Access the API documentation at `http://localhost:8000/docs`
   - Test endpoints directly from the Swagger UI

2. **Streamlit UI**:
   - Use the sidebar to navigate between different views (Chat, Curriculum, Progress, etc.)
   - Start by exploring the curriculum or asking questions in the chat
   - Take quizzes to test your knowledge
   - Track your progress through the modules

## Step 10: Using the Shell Script (Optional)

For convenience, you can use the included shell script to run different components:

```bash
# First make it executable
chmod +x run_finance_assistant.sh

# Then run it
./run_finance_assistant.sh
```

Choose from the options in the menu to:
1. Start API Server
2. Start Streamlit Interface
3. Run Content Ingestion
4. Run Example Chat
5. Start Complete System (API + UI)

## Troubleshooting

1. **Connection Issues with Supabase**:
   - Ensure your Supabase project is up and running
   - Verify the SUPABASE_URL and SUPABASE_KEY in your .env file
   - Check if the tables were created correctly

2. **API Key Issues**:
   - Verify your OpenAI and Anthropic API keys are valid and have sufficient credits
   - Ensure the .env file is being loaded correctly

3. **Content Ingestion Failures**:
   - Check if your PDF files are readable
   - Verify the documents directory path is correct
   - Look for error messages in the ingestion script output

4. **Import Errors**:
   - Ensure all dependencies are installed with `pip install -r requirements.txt`
   - Check if you're running the scripts from the correct directory

## Next Steps

- Add more educational content to the documents directory and re-run ingestion
- Customize the curriculum in `curriculum_roadmap.py` for your specific needs
- Extend the Streamlit UI with additional features in `streamlit_app.py`

## Support

If you encounter any issues, please check the project documentation or open an issue on the GitHub repository. 