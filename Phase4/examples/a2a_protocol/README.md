# A2A Simple Agent

A Simple System built on the [Agent-to-Agent (A2A) Protocol](https://a2a-protocol.org/).

## What is A2A?

The Agent-to-Agent (A2A) protocol is an open standard by Google that enables autonomous agents to communicate, coordinate, and collaborate securely. It provides:

- **Agent Discovery** — agents advertise capabilities via `AgentCard` at a well-known URL
- **Task Management** — create, monitor, cancel, and stream tasks between agents
- **Multiple Transports** — JSON-RPC, HTTP+JSON REST, and gRPC
- **Streaming** — real-time task updates via SSE or streaming RPCs
- **Versioning** — backward compatible v0.3 alongside v1.0

## Key Concepts

### AgentCard

A JSON document served at `/.well-known/agent-card.json` that describes an agent's identity, capabilities, skills, and supported transport interfaces. Every A2A agent must publish one.

```json
{
  "name": "Simple",
  "description": "Simple Agent",
  "version": "1.0.0",
  "capabilities": { "streaming": true, "pushNotifications": false },
  "skills": [{ "id": "healthcare_agent", "name": "Healthcare Agent", ... }],
  "supportedInterfaces": [
    { "protocolBinding": "JSONRPC", "protocolVersion": "1.0", "url": "..." },
    { "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0", "url": "..." }
  ]
}
```

### Tasks & Messages

- **Task** — a unit of work tracked by the agent, identified by a unique `taskId`
- **Message** — a message within a task's conversation history, with `role` (user/agent) and `parts` (text, file, etc.)
- **Task lifecycle**: `SUBMITTED` → `WORKING` → `COMPLETED` / `FAILED` / `CANCELED`

### Streaming

Agents that advertise `streaming: true` support real-time task updates via Server-Sent Events (SSE) on the REST transport or streaming RPCs on JSON-RPC.

### Protocol Versions

- **v0.3** — legacy protocol (backward compatible)
- **v1.0** — current protocol version

Both are served on the same endpoints with `A2A-Version` header negotiation.

## Project Structure

```
a2a_protocol/
├── server.py            # A2A server entry point (FastAPI + CLI)
├── agent_executor.py    # Custom AgentExecutor implementation
├── a2a_client.py        # Async A2A SDK client example
├── remote_client.py     # Simple sync HTTPX-based REST client
├── main.py              # Placeholder CLI entry
├── pyproject.toml       # Project config & dependencies
├── Makefile             # Convenience targets
└── README.md
```

| File | Purpose |
|---|---|
| `server.py` | FastAPI server that mounts JSON-RPC, REST, and Agent Card routes; CLI with `--host`/`--port` flags |
| `agent_executor.py` | Pluggable `AgentExecutor` — replace with your own agent logic |
| `a2a_client.py` | Reference async client using the official `a2a-sdk` client |
| `remote_client.py` | Minimal sync HTTPX wrapper for ad-hoc REST calls |
| `main.py` | Placeholder for packaged CLI entry (`a2a-healthcare` console script) |

## Setup

Requires **Python 3.14+**.

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Dependencies

- `a2a-sdk[fastapi]` — official A2A Python SDK with FastAPI/HTTP transport support
- `uvicorn[standard]` — ASGI server
- `litellm` — LLM gateway (available for agent integration)
- `aiosqlite` — async SQLite (available for persistent task storage)

## Usage

```bash
# Start server (default: 127.0.0.1:41241)
make server

# Or with custom host/port
python server.py --host 0.0.0.0 --port 8000
```

```bash
# Run the async SDK client
make client
```

### Verify the Server

```bash
# Fetch the Agent Card
curl http://localhost:8000/.well-known/agent-card.json

# Send a JSON-RPC message
curl -X POST http://localhost:8000/a2a/jsonrpc \
  -H "Content-Type: application/json" \
  -H "A2A-Version: 1.0" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tasks.send",
    "params": {
      "id": "test-123",
      "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "Hello"}]
      }
    }
  }'

# REST: send a message
curl -X POST http://localhost:8000/a2a/rest/message:send \
  -H "Content-Type: application/json" \
  -H "A2A-Version: 1.0" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "tasks.send",
    "params": {
      "id": "test-456",
      "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "Hello"}]
      }
    }
  }'
```

## Transports & Endpoints

| Transport | Endpoint | Versions |
|---|---|---|
| Agent Card | `GET /.well-known/agent-card.json` | — |
| JSON-RPC | `POST /a2a/jsonrpc` | v0.3, v1.0 |
| HTTP+JSON REST | `/a2a/rest/*` | v0.3, v1.0 |

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/a2a/rest/message:send` | Send a message |
| `POST` | `/a2a/rest/message:stream` | Send and stream response |
| `GET` | `/a2a/rest/tasks/{id}` | Get task status |
| `POST` | `/a2a/rest/tasks/{id}:cancel` | Cancel a task |
| `GET` | `/a2a/rest/tasks/{id}:subscribe` | Subscribe to task updates (SSE) |
| `GET` | `/a2a/rest/tasks` | List tasks |

Protocol version is selected via the `A2A-Version` header (`1.0` or `0.3`).

## Configuration

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `41241` | HTTP server port |

## Extending

### Custom Agent Logic

Replace `SampleAgentExecutor` with your own implementation:

```python
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Part, Role

class MyAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Your agent logic here
        result = await my_llm.invoke(context)
        await event_queue.enqueue_event(
            Message(role=Role.ROLE_AGENT, parts=[Part(text=result)])
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Handle cancellation
        pass
```

Then swap it in `server.py`:

```python
request_handler = DefaultRequestHandler(
    agent_executor=MyAgentExecutor(),
    ...
)
```

### Persistent Task Storage

Replace `InMemoryTaskStore` with `DatabaseTaskStore` (requires `a2a-sdk[sql]`):

```python
from a2a.server.tasks import DatabaseTaskStore

task_store = DatabaseTaskStore("sqlite+aiosqlite:///tasks.db")
```
