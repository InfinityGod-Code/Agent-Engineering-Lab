import json

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.chains.yt_chain import YTChain
from app.models.groq_model import GroqModel
from app.services.tool_executor import ToolExecutor

app = FastAPI()

model = GroqModel()
tools = ToolExecutor.get_all_tools()
chain = YTChain(tools=tools, model=model)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def read_root():
    return FileResponse("app/static/index.html")


@app.get("/summarize")
def summarize(url: str):
    summary = chain.run(url)
    return {"url": url, "summary": summary}


@app.get("/summarize/stream")
async def summarize_stream(url: str):
    async def sse_streamer():
        async for event in chain.stream_run(url):
            yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"

    return StreamingResponse(
        sse_streamer(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
