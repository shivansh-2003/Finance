---
noteId: "c62dbcc004c511f0b66a37f0c042079e"
tags: []

---

# Personal Finance Teaching Assistant - Quick Start Guide

This quick start guide will help you get the Personal Finance Teaching Assistant up and running as fast as possible.

## 1. Setup Environment

```bash
# Clone repository (if using git)
git clone https://github.com/yourusername/finance-teaching-assistant.git
cd finance-teaching-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

## 2. Configure API Keys

Edit the `.env` file and add your API keys:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## 3. Run All-In-One Script (Recommended)

The easiest way to get started is to use the provided shell script:

```bash
chmod +x run_finance_assistant.sh
./run_finance_assistant.sh
```

Then choose option 5 to "Start Complete System (API + UI)".

## 4. Manual Startup (Alternative)

If you prefer to run components manually, use these commands:

**Terminal 1 - API Server:**
```bash
cd Finace_edu
python main.py
```

**Terminal 2 - Streamlit UI:**
```bash
cd Finace_edu
streamlit run streamlit_app.py
```

## 5. Access the Application

- Streamlit UI: http://localhost:8501
- API Documentation: http://localhost:8000/docs

## 6. Sample Questions to Try

Once the application is running, try asking these questions in the chat:
- "What is compound interest?"
- "How do I create a budget?"
- "Explain the difference between a Roth IRA and a Traditional IRA"
- "How much should I save for retirement?"
- "What's a good strategy for paying off debt?"

## Need More Help?

For more detailed instructions, see the [SETUP_GUIDE.md](SETUP_GUIDE.md) file. 