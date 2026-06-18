import os
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage # Added this import

# 1. AUTHENTICATION
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 2. RAG COMPONENTS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="realtime_news",
    embedding_function=embeddings,
    persist_directory="./news_storage"
)

# 3. TOOL DEFINITIONS
@tool
def search_latest_news(query: str):
    """Searches the live web for the latest news using Tavily."""
    from langchain_community.tools.tavily_search import TavilySearchResults
    search = TavilySearchResults(k=3, tavily_api_key=TAVILY_API_KEY)
    return search.run(query)

@tool
def save_news_to_rag(text: str):
    """Splits and indexes news text into ChromaDB for RAG retrieval."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    vector_store.add_documents(docs)
    return f"Indexed {len(docs)} news chunks."

@tool
def retrieve_news_memory(query: str):
    """Retrieves news context from the vector store for accurate answering."""
    results = vector_store.similarity_search(query, k=2)
    return "\n---\n".join([r.page_content for r in results])

# 4. AGENT SETUP
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Instructions passed as a SystemMessage object to avoid keyword errors
system_msg = SystemMessage(content=(
    "You are an Advanced News Research Agent. "
    "1. Use 'search_latest_news' for current events. "
    "2. Use 'save_news_to_rag' to save facts. "
    "3. Use 'retrieve_news_memory' for history. "
    "Always include a 'Sources' section with links."
))

checkpointer = InMemorySaver()

# REMOVED state_modifier entirely to stop the TypeError.
# We pass the system_msg directly in the invoke call instead.
agent_executor = create_react_agent(
    model=model,
    tools=[search_latest_news, save_news_to_rag, retrieve_news_memory],
    checkpointer=checkpointer
)

# 5. GRADIO WRAPPER FUNCTION
def gradio_ask(message, history):
    config = {"configurable": {"thread_id": "fixed_session_v3"}}
    
    try:
        # We manually prepend the SystemMessage to the inputs here
        inputs = {"messages": [system_msg, ("user", message)]}
        
        response = agent_executor.invoke(inputs, config=config)
        
        return response["messages"][-1].content
    except Exception as e:
        return f"Deployment Error: {str(e)}"

# 6. LAUNCH INTERFACE
demo = gr.ChatInterface(
    fn=gradio_ask, 
    title="🚀 Real-Time News Agent",
    description="Agent using LangGraph, Gemini, and ChromaDB."
)

if __name__ == "__main__":
    demo.launch()
