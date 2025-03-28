from pathlib import Path
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from mygraph import build_report, invoke_graph
from utils import append_viz_to_report, exec_viz_code, move_static_files
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

app = FastAPI()

# Mount the static directory for serving images
app.mount("/static", StaticFiles(directory="./static"), name="static")


class ChatRequest(BaseModel):
    message: str
    active_agents: list[str]
    year: Optional[int]
    quarter: Optional[int]


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
def chat(request: ChatRequest):
    # Get the response from the graph
    judge_model = init_chat_model(
        model_provider="openai", model="gpt-3.5-turbo"
    )
    structured_llm = judge_model.with_structured_output(ValidationResponse)
    system_instructions = ("You are a query validator. You will be given a text and you will need to validate if it is related Nvidia. "
                           "If the text is not related to Nvidia, you will return 'invalid'. If the query is related to Nvidia, you will return 'valid'.")
    messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=request.message),
    ]
    results = structured_llm.invoke(messages)
    if results.valid == "invalid":
        return ChatResponse(report="The query is not related to Nvidia. Please provide a valid query.")
    out, viz = invoke_graph(
        request.message,
        request.year if request.year else None,
        request.quarter if request.quarter else None,
    )
    report = build_report(out)

    try:
        for viz_code, viz_reasoning in viz:
            exec_viz_code(viz_code)
        move_static_files()
        report = append_viz_to_report(report)
        return ChatResponse(report=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating visualization") from e
