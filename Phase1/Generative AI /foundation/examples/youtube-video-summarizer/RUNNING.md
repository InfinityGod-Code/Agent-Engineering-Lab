# YouTube Video Summarizer

An AI-powered YouTube video summarizer built with FastAPI, LangChain, and Groq. Extracts video transcripts and generates concise summaries using LLM inference.

## Architecture

```
FastAPI  →  LangChain LCEL Chain  →  Groq LLM (openai/gpt-oss-20b)
              ↓
          Tools: extract_video_id, get_youtube_transcript
              ↓
          Response: JSON summary or SSE stream
```

## Quick Start (Docker)

```bash
# 1. Set your Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 2. Build and run
docker compose up -d

# 3. Open http://localhost:8000
```

## Manual Setup

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### With uv

```bash
# 1. Install dependencies
uv sync

# 2. Set your Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 3. Run the server
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Open http://localhost:8000
```

### Without uv

```bash
# 1. Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi[standard] langchain langchain-core langchain-groq youtube-transcript-api pydantic-settings

# 2. Set your Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 3. Run the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Open http://localhost:8000
```

## API

### `GET /`

Serves the frontend — a single-page HTML interface. Enter a YouTube URL and click "Summarize" to see tool progress and a streaming word-by-word summary.

### `GET /summarize?url=<youtube_url>`

Returns the summary as a JSON object.

**Request:**
```
GET /summarize?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

**Response:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "summary": "The video is the official music video for Rick Astley's 1987 hit..."
}
```

### `GET /summarize/stream?url=<youtube_url>`

Streams the summarization process in real-time via Server-Sent Events. Shows tool execution progress followed by a word-by-word summary.

**Request:**
```
GET /summarize/stream?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

**Response (SSE event stream):**
```
event: status
data: {"message": "Analyzing video URL..."}

event: tool_start
data: {"name": "extract_video_id", "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}}

event: tool_end
data: {"name": "extract_video_id", "result": "dQw4w9WgXcQ"}

event: status
data: {"message": "Fetching transcript..."}

event: tool_start
data: {"name": "get_youtube_transcript", "args": {"video_id": "dQw4w9WgXcQ"}}

event: tool_end
data: {"name": "get_youtube_transcript", "result": "[\u266a\u266a\u266a]\n\u266a We're no strangers..."

event: status
data: {"message": "Generating summary..."}

event: token
data: "**"

event: token
data: "Summary"

event: token
data: " of"

...
```

### SSE Event Types

| Event        | When                          | Data                                      |
|-------------|-------------------------------|-------------------------------------------|
| `status`    | Phase change (loading state)  | `{ "message": "..." }`                    |
| `tool_start` | LLM initiates a tool call     | `{ "name": "...", "args": {...} }`        |
| `tool_end`   | Tool execution completes      | `{ "name": "...", "result": "..." }`      |
| `token`     | Summary token generated       | `"text fragment"` (raw string)            |
| `done`      | Summary complete              | `{}`                                      |

## Environment Variables

| Variable        | Required | Description                    |
|----------------|----------|--------------------------------|
| `GROQ_API_KEY` | Yes      | API key for Groq LLM inference |

## Project Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── RUNNING.md
├── main.py                  # FastAPI app with all endpoints
├── pyproject.toml
├── uv.lock
├── .env                     # API key (not committed)
├── .dockerignore
└── app/
    ├── __init__.py
    ├── config.py             # Pydantic settings (reads .env)
    ├── chains/
    │   └── yt_chain.py       # LangChain LCEL chain + stream_run
    ├── models/
    │   └── groq_model.py     # ChatGroq wrapper
    ├── services/
    │   └── tool_executor.py  # Tool dispatch and execution
    ├── static/
    │   └── index.html        # Frontend (vanilla JS + SSE)
    └── tools/
        ├── extract_video_id.py
        └── youtube_transcript.py
```

## Troubleshooting

| Problem                     | Likely Cause                        | Fix                                    |
|-----------------------------|-------------------------------------|----------------------------------------|
| Empty or error response     | API key missing or invalid          | Check `GROQ_API_KEY` in `.env`         |
| `ModuleNotFoundError`       | Dependencies not installed          | Run `uv sync` or `pip install -r ...`  |
| Docker build fails (`uv`)   | Python 3.14 image unavailable       | Change `FROM python:3.14-slim` to `python:3.13-slim` |
| "No transcript found"       | Video has no captions               | Try a different video                  |
| Connection refused          | Server not running                  | Run `docker compose up -d` or `uvicorn main:app` |
