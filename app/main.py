from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Starter LLM Service")


class EchoRequest(BaseModel):
    prompt: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/echo")
def chat_echo(payload: EchoRequest) -> dict[str, str]:
    return {"response": payload.prompt}
