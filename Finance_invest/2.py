from typing import Union, List, Tuple
from sentence_transformers import SentenceTransformer
from agno.models.groq import Groq
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
class DocumentQA:
    def __init__(self):
        # Initialize embedder
        self.embedder = self._create_embedder()
        # Initialize Groq model
        self.chat_model = Groq(id="llama3-8b-8192")
        # Database URL
        self.db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
        self.current_knowledge_base = None
        self.agent = None

    def _create_embedder(self):
        """Create the embedding model"""
        class EmbeddingModel:
            def __init__(self):
                self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
                self.dimensions = 384

            def get_embedding_and_usage(self, text: Union[str, List[str]]) -> Tuple[Union[List[List[float]], List[float]], dict]:
                if isinstance(text, str):
                    embedding = self.model.encode(text)
                    embedding_list = embedding.tolist()
                    usage = {"prompt_tokens": len(text.split()), "total_tokens": len(text.split())}
                    return embedding_list, usage
                else:
                    embeddings = self.model.encode(text)
                    embedding_list = embeddings.tolist()
                    total_tokens = sum(len(t.split()) for t in text)
                    usage = {"prompt_tokens": total_tokens, "total_tokens": total_tokens}
                    return embedding_list, usage

            def get_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
                if isinstance(text, str):
                    return self.model.encode(text).tolist()
                return self.model.encode(text).tolist()

        print("✅ Embedding model(sentence-transformers/paraphrase-MiniLM-L6-v2) initialized successfully!")
        return EmbeddingModel()

    def load_pdf_url(self, url: str, table_name: str = "documents"):
        """Load a PDF from a URL"""
        try:
            # Create PDF URL knowledge base
            self.current_knowledge_base = PDFUrlKnowledgeBase(
                urls=[url],
                vector_db=PgVector(
                    table_name=table_name,
                    db_url=self.db_url,
                    embedder=self.embedder
                ),
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
            print("✅ Knowledge base loaded successfully!")

            # Show sample content
            self.show_sample_content()

        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            import traceback
            print(traceback.format_exc())

    def show_sample_content(self, num_samples: int = 5):
        """Show sample content from the knowledge base"""
        try:
            if not self.current_knowledge_base:
                print("No knowledge base loaded!")
                return

            docs = self.current_knowledge_base.search("")
            print("\nSample documents in knowledge base:")
            print("-" * 50)
            for i, doc in enumerate(docs[:num_samples], 1):
                print(f"\nDocument {i}:")
                if hasattr(doc, 'content'):
                    print(doc.content[:200] + "..." if len(doc.content) > 200 else doc.content)
                elif hasattr(doc, 'text'):
                    print(doc.text[:200] + "..." if len(doc.text) > 200 else doc.text)
        except Exception as e:
            print(f"Error showing samples: {e}")

    def ask(self, question: str):
        """Ask a question about the loaded document"""
        if not self.current_knowledge_base or not self.agent:
            print("Please load a document first!")
            return

        print(f"\nQ: {question}")
        try:
            # Get relevant documents
            relevant_docs = self.current_knowledge_base.search(question)
            print("\nRelevant documents found:", len(relevant_docs) if relevant_docs else 0)

            # Build context from relevant documents
            context = "\n".join([doc.content if hasattr(doc, 'content') else doc.text
                               for doc in relevant_docs])

            # Create a prompt that includes the context
            full_prompt = f"""Based on the following content:{context}
            Question: {question}
            Please provide a detailed answer based ONLY on the information provided above."""

            # Get response with context
            response = self.agent.run(full_prompt)
            print(f"\nA: {response.content}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            print(traceback.format_exc())
RAG_qa = DocumentQA()
RAG_qa.ask("Key points in this report? Give in 5 bullets")            