import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Optional, TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from tavily import TavilyClient
from pinecone import Pinecone
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from snowflake_agent.sql_agent import SnowflakeAgent
from langchain_core.tools import BaseTool, Tool

load_dotenv("../.env")
client = TavilyClient(os.getenv("TAVILY_API_KEY"))


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    visualization_code: str
    visualization_reasoning: str
    snowflake_result: str
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


viz = []


# define a function to transform intermediate_steps from list
# of AgentAction to scratchpad string
def create_scratchpad(intermediate_steps: list[AgentAction]):
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            # this was the ToolExecution
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)


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
def search_pinecone(
    query: str,
    year: Optional[str] = None,
    quarter: Optional[str] = None,
    top_k: int = 2,
) -> str:
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
        filter=filter_dict if filter_dict else None,
    )

    print(
        f"Raw Pinecone response: {response}"
    )  # for debugging. can be commented out in final

    matches = response.get("matches", [])
    if not matches:
        return "No relevant documents found in Pinecone."

    context = format_rag_contexts(matches)
    print(
        f"Formatted context:\n{context}\n"
    )  # for debugging. can be commented out in final
    return context


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
        query=query, max_results=3, time_range="week", include_answer="basic"
    )

    results = response["results"]
    web_agent_contexts = "\n---\n".join(
        ["\n".join([x["title"], x["url"], x["content"]]) for x in results]
    )

    return web_agent_contexts


snowflake_agent = SnowflakeAgent(os.environ.get("SNOWFLAKE_URI"))


@tool("snowflake_agent")
def snowflake_tool(query: str):
    """
    Performs a SQL query on the Snowflake database and formats the result.
    Args:
    query (str): The SQL query to execute.
    Returns:
    str: A formatted string containing the result of the SQLQuery result, python code for visualization and reasoning for the visualization.
    """
    result = snowflake_agent.generate_query_result(query)

    # Check if visualization is needed
    visualization_info = snowflake_agent.choose_visualization(query, result)
    if visualization_info["visualization"] != "none":
        visualization = snowflake_agent.generate_visualization_code(query)
        viz.append(
            (
                visualization["visualization_code"],
                visualization["visualization_reasoning"],
            )
        )
        result = f"\n\nSqlQuery result:{result}\n\nVisualization Code:\n{visualization['visualization_code']}\n\nVisualization Reasoning:{visualization['visualization_reasoning']}"

    return result


@tool("final_answer")
def final_answer(
    introduction: str,
    research_steps: str,
    main_body: str,
    conclusion: str,
    sources: str,
    visualization_code: str,
    visualization_reasoning: str,
):
    """Returns a natural language response to the user in the form of a research
    report. There are several sections to this report, those are:
    - `introduction`: a short paragraph introducing the user's question and the
    topic we are researching.
    - `research_steps`: a few bullet points explaining the steps that were taken
    to research your report.
    - `main_body`: this is where the bulk of high quality and concise
    information that answers the user's question belongs. It is 5-6 paragraphs
    long in length.
    - `conclusion`: this is a short single paragraph conclusion providing a
    concise but sophisticated view on what was found.
    - `sources`: a bulletpoint list provided detailed sources for all information
    referenced during the research process
    - `visualization_code`: python code for visualization
    - `visualization_reasoning`: reasoning for the visualization
    """
    if isinstance(research_steps, list):
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if isinstance(sources, list):
        sources = "\n".join([f"- {s}" for s in sources])
    return ""


system_prompt = """You are the oracle, the great AI decision maker.
Given the user's query you must decide what to do with it based on the
list of tools provided to you.

If you see that a tool has been used (in the scratchpad) with a particular
query, do NOT use that same tool with the same query again. Also, do NOT use
any tool more than twice (ie, if the tool appears in the scratchpad twice, do
not use it again).

You should aim to run all the tools at least once to extract as much information as possible.
Once you have collected plenty of information
to answer the user's question in great detail(stored in the scratchpad) use the final_answer
tool."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("assistant", "scratchpad: {scratchpad}"),
    ]
)

from langchain_core.messages import ToolCall, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

# from snowflake_agent.sql_agent import SnowflakeAgent

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     api_key=SecretStr(os.environ["GEMINI_API_KEY"]),
# )
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
)

tools: list[BaseTool] = [
    snowflake_tool,
    web_search,
    search_pinecone,
    final_answer,
]


oracle = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "scratchpad": lambda x: create_scratchpad(
            intermediate_steps=x["intermediate_steps"]
        ),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)


# def run_snowflake(state: AgentState):
#     # Extract the query from the state
#     tool_name = state["intermediate_steps"][-1].tool
#     tool_args = state["intermediate_steps"][-1].tool_input

#     # Get the query text
#     query = tool_args
#     if isinstance(tool_args, dict) and "__arg1" in tool_args:
#         query = tool_args["__arg1"]

#     print(f"snowflake_agent.generate_query_result(input='{query}')")

#     # Run the query
#     result = snowflake_agent.generate_query_result(query)

#     # Check if visualization is needed
#     visualization_info = snowflake_agent.choose_visualization(query, result)
#     if visualization_info["visualization"] != "none":
#         visualization_code = snowflake_agent.generate_visualization_code(query)[
#             "visualization_code"
#         ]
#         state["visualization_code"] = visualization_code
#         state["visualization_reasoning"] = visualization_info["visualization_reasoning"]

# # Create action output
# action_out = AgentAction(tool=tool_name, tool_input=tool_args, log=str(result))
# return {"intermediate_steps": [action_out]}


def run_oracle(state: AgentState):
    print("run_oracle")
    print(f"intermediate_steps: {state['intermediate_steps']}")
    try:
        out = oracle.invoke(state)

        # Check if tool_calls exists and is not empty
        if hasattr(out, "tool_calls") and out.tool_calls:
            tool_name = out.tool_calls[0]["name"]
            tool_args = out.tool_calls[0]["args"]
            action_out = AgentAction(tool=tool_name, tool_input=tool_args, log="TBD")
            return {"intermediate_steps": [action_out]}
        else:
            # Default to final_answer if no tool calls are made
            print("No tool calls found in LLM response, defaulting to final_answer")
            action_out = AgentAction(
                tool="final_answer",
                tool_input={
                    "introduction": "Error occurred",
                    "research_steps": "The AI system encountered an error.",
                    "main_body": "Unable to process your request due to a technical issue.",
                    "conclusion": "Please try again with a different query.",
                    "sources": "Internal system error log",
                },
                log="TBD",
            )
            return {"intermediate_steps": [action_out]}
    except Exception as e:
        print(f"Error in run_oracle: {str(e)}")
        # Provide a fallback action when an error occurs
        action_out = AgentAction(
            tool="final_answer",
            tool_input={
                "introduction": "Error occurred",
                "research_steps": "The AI system encountered an error.",
                "main_body": f"Unable to process your request due to: {str(e)}",
                "conclusion": "Please try again with a different query.",
                "sources": "Internal system error log",
            },
            log="TBD",
        )
        return {"intermediate_steps": [action_out]}


def router(state: AgentState):
    # Check if intermediate_steps exists and is not empty
    if isinstance(state["intermediate_steps"], list) and state["intermediate_steps"]:
        return state["intermediate_steps"][-1].tool
    else:
        # Default to final_answer if there's an issue
        print("Router invalid format or empty intermediate steps")
        return "final_answer"


tool_str_to_func = {
    "search_pinecone": search_pinecone,
    "web_search": web_search,
    "final_answer": final_answer,
    "snowflake_agent": snowflake_tool,
}


def run_tool(state: AgentState):
    # use this as helper function so we repeat less code
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    print(f"{tool_name}.invoke(input={tool_args})")
    # run tool
    out = tool_str_to_func[tool_name].invoke(input=tool_args)
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log=str(out))
    return {"intermediate_steps": [action_out]}


def invoke_graph(input: str):
    graph = StateGraph(AgentState)

    graph.add_node("oracle", run_oracle)
    graph.add_node("snowflake_agent", run_tool)
    graph.add_node("search_pinecone", run_tool)
    graph.add_node("web_search", run_tool)
    graph.add_node("final_answer", run_tool)

    graph.set_entry_point("oracle")

    graph.add_conditional_edges(
        source="oracle",  # where in graph to start
        path=router,  # function to determine which node is called
    )

    # create edges from each tool back to the oracle
    for tool_obj in tools:
        if tool_obj.name != "final_answer":
            graph.add_edge(tool_obj.name, "oracle")

    # if anything goes to final answer, it must then move to END
    graph.add_edge("final_answer", END)

    runnable = graph.compile()
    # with open("graph.md", "w") as f:
    #     f.write(runnable.get_graph().draw_mermaid())

    oracle_out = runnable.invoke(
        {
            "input": input,
            "chat_history": [],
            "intermediate_steps": [],
        }
    )
    final_step = oracle_out["intermediate_steps"][-1]
    final_input = final_step.tool_input if isinstance(final_step, AgentAction) else {}
    return oracle_out, viz


# print(out)


def build_report(output: dict) -> str:
    """
    Build a report from the output of the graph.
    Returns a tuple of the report and the visualization code.
    """
    output = output["intermediate_steps"][-1].tool_input
    research_steps = output["research_steps"]
    if type(research_steps) is list:
        research_steps = "\n\n".join([f"- {r}" for r in research_steps])
    sources = output["sources"]
    if type(sources) is list:
        sources = "\n\n".join([f"- {s}" for s in sources])
    return f"""
## INTRODUCTION
---------------
{output['introduction']}

## RESEARCH STEPS
------------------
{research_steps}

## REPORT
--------
{output["main_body"]}

## CONCLUSION
------------
{output["conclusion"]}

SOURCES
-------
{sources}
"""


if __name__ == "__main__":
    out, viz = invoke_graph(
        "What was the trend of valuation metrics for Nvidia over past 4 quarters. Give analysis of trend and possible reasons for the trend."
    )
    report, visualization_code = build_report(out)
    exec(visualization_code)
