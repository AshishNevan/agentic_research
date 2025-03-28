from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from mygraph import build_report, invoke_graph
from fastapi.responses import Response
import os

app = FastAPI()

# Mount the static directory for serving images
app.mount("/static", StaticFiles(directory="./static"), name="static")


class ChatRequest(BaseModel):
    message: str
    active_agents: list[str]
    year: str
    quarter: str


class ChatResponse(BaseModel):
    report: str
    viz: list[tuple[str, str]]


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Get the response from the graph
    out, viz = invoke_graph(
        f"{request.message} for Year: {request.year} and Quarter: {request.quarter}"
    )
    report = build_report(out)
    print("report: ", report)
    print("viz: ", len(viz))
    # if len(viz) > 0:
    #     for id, (viz_code, viz_reasoning) in enumerate(viz):
    #         try:
    #             exec(viz_code)
    #         except Exception as e:
    #             print(f"Error executing visualization code {id}: {e}")

    # Create static directory if it doesn't exist
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    os.makedirs(static_dir, exist_ok=True)

    # Move PNG files to static directory and update markdown with correct URLs
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    png_files = [f for f in os.listdir(backend_dir) if f.lower().endswith(".png")]

    # Append images to the markdown output with static URLs
    if png_files:
        report += "\n\n## Generated Charts\n"
        for png_file in png_files:
            # Move file to static directory if it's not already there
            src_path = os.path.join(backend_dir, png_file)
            dst_path = os.path.join(static_dir, png_file)
            if not os.path.exists(dst_path):
                os.rename(src_path, dst_path)
            # Use static URL in markdown
            report += f"\n![{png_file}](/static/{png_file})"
    return ChatResponse(report=report, viz=viz)
