---
noteId: "caba7b6004c611f0b66a37f0c042079e"
tags: []

---

# Getting Started with the Personal Finance Teaching Assistant

Welcome to the Personal Finance Teaching Assistant! This document will help you navigate the documentation and get started with the system.

## Available Documentation

This project includes several documentation files to help you set up, understand, and use the system:

1. **[README.md](README.md)** - Overview of the project, its components, and features
2. **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed step-by-step setup instructions
3. **[QUICK_START.md](QUICK_START.md)** - Quick start guide for those who want to get running fast
4. **[SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)** - Technical details of the system architecture
5. **[UI_GUIDE.md](UI_GUIDE.md)** - Guide to using the Streamlit user interface

## Quick Overview

The Personal Finance Teaching Assistant is an educational system designed to help users learn about personal finance through:

- A structured 5-level curriculum from basics to advanced concepts
- Personalized teaching that adapts to your knowledge level
- Interactive quizzes and assessments
- Progress tracking and learning path recommendations
- A conversational interface to ask questions about finance topics

## Getting Started in 3 Steps

1. **Setup the System**:
   ```bash
   # Clone/download the repository
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Configure environment variables
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Run the System**:
   ```bash
   # Use the provided script
   chmod +x run_finance_assistant.sh
   ./run_finance_assistant.sh
   ```
   
   Then choose option 5 to "Start Complete System (API + UI)"

3. **Start Learning**:
   - Open the Streamlit UI at http://localhost:8501
   - Explore the curriculum or ask questions in the chat
   - Take quizzes to test your knowledge
   - Track your progress through the modules

## Next Steps

After getting the system running:

1. **Add Educational Content**: Put PDF files in the `documents` directory and run the ingestion process
2. **Customize the Curriculum**: Modify `curriculum_roadmap.py` to suit your specific needs
3. **Extend the UI**: Add more features to the Streamlit interface in `streamlit_app.py`

## Need Help?

- For detailed setup instructions, refer to [SETUP_GUIDE.md](SETUP_GUIDE.md)
- For technical details about how the system works, see [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)
- For UI navigation help, check [UI_GUIDE.md](UI_GUIDE.md)

Enjoy learning about personal finance with your new AI assistant! 