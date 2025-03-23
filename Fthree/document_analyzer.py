# finance_agent/document_analyzer.py

import os
import tempfile
from typing import Dict, List, Any, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from agno.agent import Agent
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.groq import Groq

class DocumentAnalyzer:
    """
    Agent for analyzing financial documents using RAG (Retrieval Augmented Generation).
    """
    
    def __init__(self):
        # Initialize LLM and embeddings
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.2)
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize document schemas for different financial report types
        self.document_schemas = {
            "10k": self._get_10k_schema(),
            "10q": self._get_10q_schema(),
            "annual_report": self._get_annual_report_schema(),
            "earnings_call": self._get_earnings_call_schema(),
            "research_report": self._get_research_report_schema(),
        }
        
        # Create temporary directory for document storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Set up DB for RAG engine
        self.db_url = "sqlite:///finance_documents.db"
    
    def query_document(self, document_path: str, query: str) -> Dict[str, Any]:
        """
        Query a document with a specific question.
        
        Parameters:
        - document_path: Path to the document file
        - query: The question to ask about the document
        
        Returns:
        - Dictionary containing the response and context
        """
        # Determine the document type and extension
        document_extension = os.path.splitext(document_path)[1].lower()
        
        # Load the document based on file type
        documents = self._load_document(document_path, document_extension)
        
        # Split the document into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Create a vector store from the chunks
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=os.path.join(self.temp_dir, "chroma_db")
        )
        
        # Create a retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial document analysis expert. 
            Your task is to provide accurate, detailed answers to questions about financial documents.
            
            Use the following context to answer the question. If the information isn't found in the context,
            say so rather than making up an answer. If possible, cite specific sections or page numbers.
            
            Focus on providing facts and analysis, not opinions. Be concise but thorough in your response.
            
            Context:
            {context}"""),
            ("human", "{question}")
        ])
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": qa_prompt}
        )
        
        # Run the query
        result = qa_chain.invoke({"query": query})
        
        # Get the relevant document chunks used for the answer
        retrieved_docs = retriever.get_relevant_documents(query)
        relevant_contexts = [doc.page_content for doc in retrieved_docs]
        
        return {
            "response": result["result"],
            "contexts": relevant_contexts,
            "document_path": document_path
        }
    
    def summarize_document(self, document_path: str, summary_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Generate a summary of a financial document.
        
        Parameters:
        - document_path: Path to the document file
        - summary_type: Type of summary to generate (brief, comprehensive, key_points, financial_metrics, risk_factors)
        
        Returns:
        - Dictionary containing the summary
        """
        # Determine the document type and extension
        document_extension = os.path.splitext(document_path)[1].lower()
        document_filename = os.path.basename(document_path)
        
        # Try to infer document type from filename
        document_type = self._infer_document_type(document_filename)
        
        # Load the document based on file type
        documents = self._load_document(document_path, document_extension)
        
        # Split the document into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Create map reduce summary prompt based on summary type and document type
        system_prompt = self._get_summary_prompt(summary_type, document_type)
        
        # Create the map prompt
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a financial document analysis expert.
            Your task is to extract key information from a section of a financial document.
            
            {system_prompt}
            
            Focus only on the information provided in this section.
            Be accurate, concise, and factual.
            """),
            ("human", "{text}")
        ])
        
        # Create the combine prompt
        combine_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a financial document analysis expert.
            Your task is to create a coherent, well-structured summary from the extracted information from different sections.
            
            {system_prompt}
            
            Organize the information in a logical structure.
            Eliminate redundancies and ensure a smooth flow.
            Be accurate, concise, and factual.
            
            Sections:
            {{text}}
            """),
            ("human", "Create a comprehensive summary based on the sections above.")
        ])
        
        # Create the summary chain
        summary_chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True
        )
        
        # Run the summarization
        summary = summary_chain.invoke({"input_documents": chunks})
        
        # Extract key financial metrics if applicable
        financial_metrics = {}
        if document_type in ["10k", "10q", "annual_report"]:
            financial_metrics = self._extract_financial_metrics(chunks)
        
        # Extract key risks if applicable
        risks = []
        if document_type in ["10k", "10q", "annual_report"]:
            risks = self._extract_risks(chunks)
        
        return {
            "response": summary["output_text"],
            "document_type": document_type,
            "summary_type": summary_type,
            "document_path": document_path,
            "financial_metrics": financial_metrics,
            "risks": risks
        }
    
    def analyze_with_agno(self, document_path: str, query: str = None) -> Dict[str, Any]:
        """
        Analyze a document using AGNO's PDF knowledge base.
        
        Parameters:
        - document_path: Path to the document file
        - query: Optional specific query about the document
        
        Returns:
        - Dictionary containing the analysis
        """
        # Create the embedding model using AGNO
        class DocumentQA:
            def __init__(self):
                # Initialize Groq model
                self.chat_model = Groq(id="llama3-8b-8192")
                # Database URL
                self.db_url = "sqlite:///finance_documents.db"
                self.current_knowledge_base = None
                self.agent = None

            def load_pdf(self, pdf_path):
                """Load a PDF file"""
                try:
                    # Create PDF knowledge base
                    self.current_knowledge_base = PDFUrlKnowledgeBase(
                        paths=[pdf_path],
                        vector_db="chroma",
                    )

                    # Initialize the Agent
                    self.agent = Agent(
                        knowledge=self.current_knowledge_base,
                        search_knowledge=True,
                        model=self.chat_model
                    )

                    # Load knowledge base
                    print("Loading knowledge base...")
                    self.current_knowledge_base.load(recreate=True)
                    print("Knowledge base loaded successfully!")

                except Exception as e:
                    print(f"Error loading PDF: {e}")
                    raise e

            def ask(self, question):
                """Ask a question about the loaded document"""
                """Ask a question about the loaded document"""
                if not self.current_knowledge_base or not self.agent:
                    print("Please load a document first!")
                    return {"error": "No document loaded"}

                try:
                    # Get relevant documents
                    relevant_docs = self.current_knowledge_base.search(question)
                    
                    # Build context from relevant documents
                    context = "\n".join([doc.content if hasattr(doc, 'content') else doc.text
                                      for doc in relevant_docs])

                    # Create a prompt that includes the context
                    full_prompt = f"""Based on the following content from the financial document:
                    
                    {context}
                    
                    Question: {question}
                    
                    Please provide a detailed answer based ONLY on the information provided above."""

                    # Get response with context
                    response = self.agent.run(full_prompt)
                    return {"response": response.content, "context": context}

                except Exception as e:
                    print(f"Error: {e}")
                    return {"error": str(e)}

        # Initialize document QA
        doc_qa = DocumentQA()
        
        # Load the document
        doc_qa.load_pdf(document_path)
        
        # If no specific query is provided, generate a general analysis
        if not query:
            query = "Provide a comprehensive analysis of this financial document, including key findings, financial metrics, risks, and outlook."
            
        # Get the response
        result = doc_qa.ask(query)
        
        return result
    
    def _load_document(self, document_path: str, document_extension: str):
        """Load a document based on file type"""
        if document_extension == ".pdf":
            loader = PyPDFLoader(document_path)
            return loader.load()
        elif document_extension in [".csv", ".xlsx", ".xls"]:
            loader = CSVLoader(document_path)
            return loader.load()
        else:
            raise ValueError(f"Unsupported document format: {document_extension}")
    
    def _infer_document_type(self, filename: str) -> str:
        """Infer document type from filename"""
        filename_lower = filename.lower()
        
        if "10-k" in filename_lower or "10k" in filename_lower:
            return "10k"
        elif "10-q" in filename_lower or "10q" in filename_lower:
            return "10q"
        elif "annual" in filename_lower and "report" in filename_lower:
            return "annual_report"
        elif "earnings" in filename_lower and ("call" in filename_lower or "transcript" in filename_lower):
            return "earnings_call"
        elif "research" in filename_lower or "analyst" in filename_lower:
            return "research_report"
        else:
            return "general"
    
    def _get_summary_prompt(self, summary_type: str, document_type: str) -> str:
        """Get the appropriate summary prompt based on summary type and document type"""
        base_prompt = ""
        
        # Adjust based on document type
        if document_type == "10k":
            base_prompt += "This is a 10-K annual report filed with the SEC. "
        elif document_type == "10q":
            base_prompt += "This is a 10-Q quarterly report filed with the SEC. "
        elif document_type == "annual_report":
            base_prompt += "This is an annual report for shareholders. "
        elif document_type == "earnings_call":
            base_prompt += "This is an earnings call transcript. "
        elif document_type == "research_report":
            base_prompt += "This is a financial research report. "
        
        # Adjust based on summary type
        if summary_type == "brief":
            base_prompt += """
            Create a brief executive summary (2-3 paragraphs) that captures the most essential information.
            Focus on the company's financial performance, significant events, and outlook.
            """
        elif summary_type == "comprehensive":
            base_prompt += """
            Create a comprehensive summary that includes:
            1. Company overview and business description
            2. Financial performance and key metrics
            3. Business segments and their performance
            4. Management's discussion and analysis
            5. Risk factors and challenges
            6. Future outlook and guidance
            7. Notable events or changes since the last report
            
            Organize the information in a clear, logical structure with headings.
            """
        elif summary_type == "key_points":
            base_prompt += """
            Extract and list the key points from the document in bullet point format.
            Focus on facts, figures, and significant statements.
            Group similar points under appropriate headings.
            """
        elif summary_type == "financial_metrics":
            base_prompt += """
            Focus specifically on financial metrics and performance.
            Include revenue, profit margins, earnings per share, debt levels, cash flow, and other relevant financial indicators.
            Compare current figures with previous periods when available.
            Explain significant changes or trends in the financial data.
            """
        elif summary_type == "risk_factors":
            base_prompt += """
            Focus on identifying and explaining all risk factors mentioned in the document.
            Categorize risks (e.g., operational, financial, regulatory, market, competitive).
            Highlight new risks or changes to previously reported risks.
            Include management's assessment of risk impact and mitigation strategies if mentioned.
            """
        
        return base_prompt
    
    def _extract_financial_metrics(self, chunks) -> Dict[str, Any]:
        """Extract key financial metrics from document chunks"""
        # Create a specialized prompt for extracting financial metrics
        metrics_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst specializing in extracting quantitative data from financial documents.
            
            Extract ALL financial metrics mentioned in the text, including but not limited to:
            - Revenue/Sales
            - Net Income/Profit
            - Earnings Per Share (EPS)
            - Gross Margin
            - Operating Margin
            - Return on Assets (ROA)
            - Return on Equity (ROE)
            - Debt-to-Equity Ratio
            - Current Ratio
            - Cash and Cash Equivalents
            - Capital Expenditures
            - Free Cash Flow
            
            Format your response as a JSON object with metric names as keys and values that include:
            - figure: the numerical value
            - period: the time period it refers to
            - change: percentage change from previous period if available
            - context: brief context about the metric
            
            If a metric is mentioned multiple times with different values, include the most recent or relevant one.
            Only include metrics that are explicitly stated in the text with actual values.
            """),
            ("human", "{text}")
        ])
        
        # Select a subset of chunks likely to contain financial information
        financial_chunks = []
        for chunk in chunks:
            text = chunk.page_content.lower()
            if any(term in text for term in ["financial", "revenue", "income", "earnings", "profit", "margin", "balance sheet", "cash flow"]):
                financial_chunks.append(chunk)
        
        # If we have too many chunks, select the most relevant ones
        if len(financial_chunks) > 5:
            financial_chunks = financial_chunks[:5]
        
        # Combine the selected chunks
        combined_text = "\n\n".join([chunk.page_content for chunk in financial_chunks])
        
        # Extract metrics
        metrics_extraction = self.llm.invoke(metrics_prompt.format(text=combined_text))
        
        # Parse the response as JSON
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'```json\n(.*?)\n```', metrics_extraction.content, re.DOTALL)
            if json_match:
                metrics_json = json_match.group(1)
            else:
                metrics_json = metrics_extraction.content
                
            metrics = json.loads(metrics_json)
            return metrics
        except Exception as e:
            print(f"Error parsing financial metrics: {e}")
            return {"error": str(e), "raw_response": metrics_extraction.content}
    
    def _extract_risks(self, chunks) -> List[Dict[str, str]]:
        """Extract key risks from document chunks"""
        # Create a specialized prompt for extracting risks
        risks_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a risk assessment specialist analyzing financial documents.
            
            Extract ALL significant risk factors mentioned in the text. Focus on:
            - Market risks
            - Operational risks
            - Financial risks
            - Regulatory risks
            - Strategic risks
            - Competitive risks
            - Technology risks
            - Reputational risks
            
            Format your response as a JSON array of risk objects, each containing:
            - category: risk category
            - description: brief description of the risk
            - impact: potential impact if mentioned
            - mitigation: mitigation strategy if mentioned
            
            Only include risks that are explicitly stated in the text.
            """),
            ("human", "{text}")
        ])
        
        # Select chunks likely to contain risk information
        risk_chunks = []
        for chunk in chunks:
            text = chunk.page_content.lower()
            if any(term in text for term in ["risk", "uncertainty", "challenge", "threat", "factor", "contingent", "liability"]):
                risk_chunks.append(chunk)
        
        # If we have too many chunks, select the most relevant ones
        if len(risk_chunks) > 5:
            risk_chunks = risk_chunks[:5]
        
        # Combine the selected chunks
        combined_text = "\n\n".join([chunk.page_content for chunk in risk_chunks])
        
        # Extract risks
        risks_extraction = self.llm.invoke(risks_prompt.format(text=combined_text))
        
        # Parse the response as JSON
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'```json\n(.*?)\n```', risks_extraction.content, re.DOTALL)
            if json_match:
                risks_json = json_match.group(1)
            else:
                risks_json = risks_extraction.content
                
            risks = json.loads(risks_json)
            return risks
        except Exception as e:
            print(f"Error parsing risks: {e}")
            return [{"category": "error", "description": str(e), "raw_response": risks_extraction.content}]
    
    def _get_10k_schema(self) -> Dict[str, List[str]]:
        """Get the schema for 10-K reports"""
        return {
            "sections": [
                "Business",
                "Risk Factors",
                "Management's Discussion and Analysis",
                "Financial Statements",
                "Notes to Financial Statements",
                "Controls and Procedures",
                "Executive Compensation",
                "Security Ownership",
                "Related Party Transactions"
            ],
            "financial_metrics": [
                "Revenue",
                "Cost of Revenue",
                "Gross Profit",
                "Operating Expenses",
                "Operating Income",
                "Net Income",
                "Earnings Per Share",
                "Total Assets",
                "Total Liabilities",
                "Shareholders' Equity",
                "Cash and Cash Equivalents",
                "Total Debt"
            ]
        }
    
    def _get_10q_schema(self) -> Dict[str, List[str]]:
        """Get the schema for 10-Q reports"""
        return {
            "sections": [
                "Financial Statements",
                "Management's Discussion and Analysis",
                "Risk Factors",
                "Controls and Procedures",
                "Legal Proceedings",
                "Unregistered Sales of Equity Securities"
            ],
            "financial_metrics": [
                "Revenue",
                "Cost of Revenue",
                "Gross Profit",
                "Operating Expenses",
                "Operating Income",
                "Net Income",
                "Earnings Per Share",
                "Total Assets",
                "Total Liabilities",
                "Shareholders' Equity",
                "Cash and Cash Equivalents"
            ]
        }
    
    def _get_annual_report_schema(self) -> Dict[str, List[str]]:
        """Get the schema for annual reports"""
        return {
            "sections": [
                "Letter to Shareholders",
                "Company Overview",
                "Financial Highlights",
                "Business Segments",
                "Management Discussion",
                "Financial Statements",
                "Notes to Financial Statements",
                "Auditor's Report",
                "Corporate Information"
            ],
            "financial_metrics": [
                "Revenue",
                "Gross Profit",
                "Operating Income",
                "Net Income",
                "Earnings Per Share",
                "Dividends",
                "Return on Equity",
                "Total Assets",
                "Total Liabilities",
                "Shareholders' Equity"
            ]
        }
    
    def _get_earnings_call_schema(self) -> Dict[str, List[str]]:
        """Get the schema for earnings call transcripts"""
        return {
            "sections": [
                "Introduction",
                "Opening Remarks",
                "Financial Results",
                "Business Update",
                "Outlook and Guidance",
                "Q&A Session",
                "Closing Remarks"
            ],
            "topics": [
                "Revenue Growth",
                "Profit Margins",
                "New Products/Services",
                "Market Conditions",
                "Competition",
                "Strategic Initiatives",
                "Challenges",
                "Future Plans"
            ]
        }
    
    def _get_research_report_schema(self) -> Dict[str, List[str]]:
        """Get the schema for research reports"""
        return {
            "sections": [
                "Executive Summary",
                "Investment Thesis",
                "Company Overview",
                "Industry Analysis",
                "Financial Analysis",
                "Valuation",
                "Risk Assessment",
                "Recommendation"
            ],
            "topics": [
                "Business Model",
                "Competitive Position",
                "Growth Drivers",
                "Financial Performance",
                "Price Targets",
                "Bull Case",
                "Bear Case",
                "Key Metrics"
            ]
        }