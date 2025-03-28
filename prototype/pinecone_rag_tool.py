import os
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_openai import ChatOpenAI
load_dotenv()

# Initialize Pinecone and embedding model
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("document-embeddings-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Utility function to format Pinecone results
def format_rag_contexts(matches: list):
    contexts = []
    for x in matches:
        text = (
            f"Text: {x['metadata']['text']}\n"
            f"Year: {x['metadata'].get('year', 'N/A')}\n"
            f"Quarter: {x['metadata'].get('quarter', 'N/A')}\n"
        )
        contexts.append(text)
    return "\n---\n".join(contexts)

@tool("search_pinecone")
def search_pinecone(query: str, year: str = None, quarter: str = None, top_k: int = 5) -> str:
    """
    Search Pinecone for documents matching the query, optionally filtered by year and quarter.
    """
    query_emb = embeddings.embed_query(query)

    # Build optional filter
    filter_dict = {}
    if year:
        filter_dict["year"] = year
    if quarter:
        filter_dict["quarter"] = quarter

    # Run Pinecone query
    response = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )

    print(f"Raw Pinecone response: {response}")  #for debugging. can be commented out in final

    matches = response.get("matches", [])
    if not matches:
        return "No relevant documents found in Pinecone."

    context = format_rag_contexts(matches)
    print(f"Formatted context:\n{context}\n")  #for debugging. can be commented out in final
    return context


llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

agent = create_conversational_retrieval_agent(
    llm,
    tools=[search_pinecone],
    verbose=True
)

result = agent.invoke("search_pinecone with query='how is the Nvidia's performance in 2024?', top_k=5, year='', quarter=''")

print(f"result-{result}\n")