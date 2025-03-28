from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse
from typing import List, Dict
from oracle_test import compile_graph, search_pinecone, web_search  
from pinecone_rag_tool import *
from tavily_agent import tavily_agent
import markdown

app = FastAPI()

# Store latest report
latest_markdown_report = ""

class ToolRequest(BaseModel):
    tool: str
    query: str
    top_k: int = 5
    year: str = None
    quarter: str = None

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]]

@app.post("/run-tool")
def run_tool(req: ToolRequest):
    if req.tool == "search_pinecone":
        result = pinecone_agent.invoke({
    "input": f"query: {req.query}, year: {req.year}, quarter: {req.quarter}, top_k: {req.top_k}"
        })
        # result = pinecone_agent.invoke({
        #     "query": req.query,
        #     "top_k": req.top_k,
        #     "year": req.year,
        #     "quarter": req.quarter
        # })
    elif req.tool == "web_search":
        # llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        # tavily_agent = create_conversational_retrieval_agent(
        #     llm,
        #     tools=[web_search],
        #     verbose=True
        # )

        result = tavily_agent.invoke({"query": req.query})
    else:
        result = f"You ran a custom tool with query: {req.query}"

    return {"result": result}

@app.post("/chat")
def chat_with_agent(req: ChatRequest):
    runnable = compile_graph()
    result = runnable.invoke({
        "input": req.query,
        "chat_history": req.chat_history,
    })
    global latest_markdown_report
    latest_markdown_report = build_report(result["intermediate_steps"][-1].tool_input)
    return {"response": latest_markdown_report}

@app.get("/download-report")
def download_report():
    global latest_markdown_report
    if latest_markdown_report:
        with open("nvidia_report.md", "w", encoding="utf-8") as f:
            f.write(latest_markdown_report)
        return FileResponse("nvidia_report.md", media_type="text/markdown", filename="nvidia_report.md")
    return {"error": "No report available yet."}

# Reuse your existing report builder

def build_report(output: dict):
    research_steps = output["research_steps"]
    if isinstance(research_steps, list):
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    sources = output["sources"]
    if isinstance(sources, list):
        sources = "\n".join([f"- {s}" for s in sources])

    return f"""
INTRODUCTION
------------
{output['introduction']}

RESEARCH STEPS
--------------
{research_steps}

REPORT
------
{output['main_body']}

CONCLUSION
----------
{output['conclusion']}

SOURCES
-------
{sources}
"""
