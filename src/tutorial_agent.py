from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import TokenTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

def create_tutorial_agent(model,embeddings_model, vectorstore):
    """
    Create an agnet to recommend a move by doing RAG from vectorstore

    Args:
        model: The language model to use for the agent
        embeddings_model: The embeddings model for vector operations
        vectorstore: The vector database containing chess moves
        
    Returns:
        A Rag agent configured with the provided resources
    """
    google_search = TavilySearchResults(max_results=10)
    config = {"configurable": {"thread_id": "searchmemthread"} }

    @tool
    def query_knowledge_base(query:str) -> list:
        """
        Get relevant documents on a relevant vectorstore based on user query
        """
        docs = vectorstore.similarity_search(query, k=5)
        return [doc.page_content for doc in docs]
        
    @tool
    def wikipedia_search(query: str) -> str:
        """
        Query Wikipedia for information related to the user's question, good for factual info such as chess events and player biographies
        """
        docs = WikipediaLoader(query=query, load_max_docs=3).load()
        # Optionally split or truncate text if needed
        splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)
        return "\n\n".join(doc.page_content for doc in split_docs)

    tools = [google_search, wikipedia_search, query_knowledge_base]

    prompt = """
    You are an intelligent agent that answer questions about the chess using retrieved documents.
    Follow these steps to answer user queries:
    1. Use the `query_knowledge_base` tool to retrieve relevant documents based on the user query.
    2. Use the `google_search` tool to retrieve relevant documents based on the user query.
    3. Use the `wikipedia_search` tool to retrieve relevant documents based on the user query.
    4. If you can't find enough information, state that explicitly.

    Use a step-by-step approach to complete each query.
    """

    tutorial_agent = create_react_agent(model, tools, prompt = prompt)

    return tutorial_agent





