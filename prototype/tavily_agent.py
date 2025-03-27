import os
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()
from tavily import TavilyClient
client = TavilyClient(os.getenv("TAVILY_API_KEY"))


@tool("web_search")
def web_search(query: str):
    """
    Perform a web search using the Tavily API and format the results.

    Args:
    query (str): The search query to execute.

    Returns:
    str: A formatted string containing the titles, URLs, and content of search results.
    """
    response = client.search(
        query=query,
        max_results=3,
        time_range="week",
        include_answer="basic"
    )


    results = response['results']
    web_agent_contexts = "\n---\n".join(
        ["\n".join([x["title"], x["url"], x["content"]]) for x in results]  )
    
    return web_agent_contexts



llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
#search = TavilySearchAPIWrapper()
#tavily_tool = TavilySearchResults(api_wrapper=search)

# Set up the agent
agent = create_conversational_retrieval_agent(
    llm,
    tools=[web_search],
    verbose=True
)

#Example usage
result = agent.invoke("web_search with query'how is the Nvidia's performance in 2024?'")
print(result)
