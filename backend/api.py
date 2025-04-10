from typing import Optional, Literal
from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from mygraph import build_report, graph_with_tools
from utils import append_viz_to_report, exec_viz_code, move_static_files
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI()

# Mount the static directory for serving images
app.mount("/static", StaticFiles(directory="./static"), name="static")


class ChatRequest(BaseModel):
    message: str
    snowflake_agent: bool = True
    pinecone_agent: bool = True
    websearch_agent: bool = True
    year: Optional[int] = None
    quarter: Optional[int] = None


class ChatResponse(BaseModel):
    report: str


class ValidationResponse(BaseModel):
    valid: Literal["valid", "invalid"] = Field(
        description="Evaluation result indicating whether the job description is valid or not."
    )


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest = Depends()):
    # Get the response from the graph
    if not (
        request.snowflake_agent or request.websearch_agent or request.pinecone_agent
    ):
        return HTTPException(400, "Select at least one agent")
    judge_model = init_chat_model(model_provider="openai", model="gpt-3.5-turbo")
    structured_llm = judge_model.with_structured_output(ValidationResponse)
    system_instructions = (
        "You are a query validator. You will be given a text and you will need to validate if it is related Nvidia. "
        "If the text is not related to Nvidia, you will return 'invalid'. If the query is related to Nvidia, you will return 'valid'."
    )
    messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=request.message),
    ]
    results = structured_llm.invoke(messages)
    if results.valid == "invalid":
        return ChatResponse(
            report="The query is not related to Nvidia. Please provide a valid query."
        )
    compiled_graph = graph_with_tools(
        snowflake=request.snowflake_agent,
        pinecone=request.pinecone_agent,
        websearch=request.websearch_agent,
    )
    out = compiled_graph.invoke(
        {
            "input": request.message,
            "year": request.year if request.year else None,
            "quarter": request.quarter if request.quarter else None,
            "chat_history": [],
            "intermediate_steps": [],
        }
    )
    report, viz = build_report(out)

    try:
        for viz_code, viz_reasoning in viz:
            print(viz_code)
            exec_viz_code(viz_code)
        move_static_files()
        report = append_viz_to_report(report)
        return ChatResponse(report=report)
    except Exception as e:
        print(e)
