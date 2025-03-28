from fastapi import FastAPI
from pydantic import BaseModel
from 

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chat")
async def chat(request: ChatRequest):
    

