from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from mygraph import build_report, invoke_graph
from fastapi.responses import Response

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chat", response_class=FileResponse)
def chat(request: ChatRequest):
    out = build_report(invoke_graph(request.message))

    # Return the markdown content directly without writing to filesystem
    return Response(
        content=out,
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=report.md"},
    )
