# Agent Tools Demo

A streaming LLM agent demo using FastAPI, LangChain, and Groq. The agent can use tools (get weather, calculator) and streams responses in real-time via Server-Sent Events.

## Architecture

```
FastAPI  →  LangChain Agent  →  Groq LLM (llama-3.3-70b-versatile)
              ↓
          Tools: get_weather, calculator
              ↓
          SSE stream → Browser (vanilla JS)
```

## Quick Start (Docker)

```bash
# 1. Set your Groq API key in .env
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 2. Build and run
docker compose up -d

# 3. Open http://localhost:8000
```

## Manual Setup

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (package manager)

### Steps

```bash
# 1. Clone and enter the project
cd agent-tools-demo

# 2. Install dependencies
uv sync

# 3. Set your Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 4. Run the server
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 5. Open http://localhost:8000
```

### Without uv

```bash
pip install fastapi[standard] langchain langchain-groq pydantic-settings
echo "GROQ_API_KEY=gsk_your_key_here" > .env
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

### `GET /`
Serves the frontend (single-page HTML).

### `POST /generate`
Streams LLM response with tool calls via SSE.

**Request:**
```json
{ "prompt": "What is the weather in London?" }
```

**Response (SSE events):**
```
event: tool_start
data: {"name": "get_weather", "args": "{\"location\":\"London\"}"}

event: tool_end
data: {"name": "get_weather", "result": "The weather in London is sunny and 72°F."}

event: token
data: "The"

event: token
data: " weather"
...
```

#### Event types

| Event        | When                     | Data                                  |
|-------------|--------------------------|---------------------------------------|
| `tool_start` | LLM calls a tool         | `{ name, args }`                      |
| `tool_end`   | Tool returns result      | `{ name, result }`                    |
| `token`      | LLM generates text token | `"string token"`                      |

## Environment Variables

| Variable        | Required | Description                     |
|----------------|----------|---------------------------------|
| `GROQ_API_KEY` | Yes      | API key for Groq LLM inference  |

## Tools

- **get_weather(location)** — Returns mock weather for a city.
- **calculator(expression)** — Evaluates a math expression.

## Troubleshooting

| Problem                     | Likely Cause                         | Fix                                   |
|-----------------------------|--------------------------------------|---------------------------------------|
| Empty response / no events  | API key missing or invalid           | Check `GROQ_API_KEY` in `.env`        |
| "Error: 500"                | Groq API error or tool exception     | Check server logs                     |
| Agent loops calling a tool  | LLM not respecting stop condition    | The system prompt handles this; if it persists, try a different model via `GroqLLM(model="...")` |
| Docker build fails          | Python 3.14 image not yet available  | Change `FROM python:3.14-slim` to `FROM python:3.13-slim` in Dockerfile |
