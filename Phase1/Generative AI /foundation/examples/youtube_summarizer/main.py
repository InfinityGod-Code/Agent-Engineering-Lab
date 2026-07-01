from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.models.base_model import BaseModel
from app.models.groq_model import GroqLLM

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def read_root():
    return FileResponse("app/static/index.html")


@app.post("/generate")
async def generate(prompt: str = Body(..., embed=True)):
    base_model = BaseModel(llm=GroqLLM())
    
    # 1. Create a wrapper generator that yields standard SSE formatted data
    async def sse_streamer():
        # base_model.stream_content must yield raw text string tokens
        async for chunk in base_model.stream_content(prompt):
            if chunk:
                yield f"data: {chunk}\n\n"

    # 2. Add performance headers to force the stream to flash real-time chunks
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    }

    return StreamingResponse(
        sse_streamer(),
        media_type="text/event-stream",
        headers=headers
    )