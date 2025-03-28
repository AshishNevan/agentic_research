import os
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()

# set up API key
# os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# # set up the agent
# llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
# search = TavilySearchAPIWrapper()
# tavily_tool = TavilySearchResults(api_wrapper=search)

# # initialize the agent
# agent = create_conversational_retrieval_agent(
#     llm,
#     tools=[tavily_tool],
#     verbose=True,
# )
# # run the agent
# agent.invoke("What is the latest news about Nvidia")


from tavily import TavilyClient
client = TavilyClient(os.getenv("TAVILY_API_KEY"))
response = client.search(
    query="What is the latest news about Nvidia?",
    max_results=2,
    time_range="week",
    include_answer="basic"
)
#print(response)

extracted_data_context = []
for result in response['results']:
    extracted_info = {
        'title': result['title'],
        'url': result['url'],
        'content': result['content']
    }
    extracted_data_context.append(extracted_info)

#Printing the extracted data
for info in extracted_data_context:
    print(f"Title: {info['title']}\nURL: {info['url']}\nContent: {info['content']}\n")

print(f" extracted data{extracted_data_context}\n")
#print(f"extracted info{extracted_info}")


