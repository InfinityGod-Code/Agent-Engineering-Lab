import json

from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.models.agent_model import AgentModel
from app.models.groq_model import GroqLLM
from app.tools.dummy_tools import dummy_tools

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def read_root():
    return FileResponse("app/static/index.html")


@app.post("/generate")
async def generate(prompt: str = Body(..., embed=True)):
    agent_model = AgentModel(llm=GroqLLM())

    async def sse_streamer():
        async for event in agent_model.stream_with_agent(
            prompt,
            tools=dummy_tools
        ):
            yield (
                f"event: {event['type']}\n"
                f"data: {json.dumps(event['data'])}\n\n"
            )

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        sse_streamer(),
        media_type="text/event-stream",
        headers=headers,
    )
