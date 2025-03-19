# import basics
import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain import hub

from supabase.client import Client, create_client
from langchain_core.tools import tool

# load environment variables
load_dotenv()  

# initiate supabase database
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# initiate vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# initiate large language model (temperature = 0)
llm = ChatOpenAI(temperature = 0) 
# fetch the prompt from the prompt hub
prompt = PromptTemplate.from_template(
    "You are a highly knowledgeable assistant specializing in personal finance. "
    "Your task is to provide clear, concise, and actionable advice based on the following question and the relevant documents: {input}. "
    "Consider the context of personal finance, including budgeting, investing, saving, and financial planning. "
    "If the answer requires specific examples or references from the documents, please include them. "
    "Use the following scratchpad for any intermediate thoughts: {agent_scratchpad}"
)

# create the tools
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    # Limit to a few documents to avoid multiple similar responses
    retrieved_docs = vector_store.similarity_search(query, k=2)  # Fetch 3 documents
    if not retrieved_docs:
        return "No relevant information found.", []

    # Serialize the content and filter for uniqueness
    unique_content = {}
    for doc in retrieved_docs:
        content = f"Source: {doc.metadata}\nContent: {doc.page_content}"
        # Use the first 100 characters as a key to filter similar responses
        key = content[:100]  
        if key not in unique_content:
            unique_content[key] = content

    # Join unique contents into a single response
    serialized = "\n\n".join(unique_content.values())
    
    # Optionally summarize the response (you can implement a summarization function here)
    # serialized = summarize_response(serialized)  # Uncomment if you have a summarization function

    return serialized, retrieved_docs

# combine the tools and provide to the llm
tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# invoke the agent
response = agent_executor.invoke({"input": "how do you determine that you are rich"})

# put the result on the screen
print("Output Type:", type(response["output"]))  # Print the type of the output
#print("Output Content:", response["output"])      # Print the actual output content