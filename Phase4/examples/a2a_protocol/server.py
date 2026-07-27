import argparse
import asyncio
import logging

import uvicorn
from fastapi import FastAPI

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
)

from agent_executor import SampleAgentExecutor

logger = logging.getLogger(__name__)


def serve(
    host: str = "127.0.0.1",
    port: int = 41241,
) -> None:
    """Run the agent with mounted JSON-RPC and HTTP+JSON transports."""
    agent_card = AgentCard(
        name="HealthcareAgent",
        description="Healthcare Triage & Coordination Agent",
        provider=AgentProvider(organization="A2A Samples", url="https://example.com"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text", "task-status"],
        skills=[
            AgentSkill(
                id="healthcare_agent",
                name="Healthcare Agent",
                description="Healthcare triage and coordination.",
                tags=["healthcare", "triage"],
                examples=["I have a fever"],
                input_modes=["text"],
                output_modes=["text", "task-status"],
            )
        ],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url=f"http://{host}:{port}/a2a/jsonrpc",
            ),
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="0.3",
                url=f"http://{host}:{port}/a2a/jsonrpc",
            ),
            AgentInterface(
                protocol_binding="HTTP+JSON",
                protocol_version="1.0",
                url=f"http://{host}:{port}/a2a/rest",
            ),
            AgentInterface(
                protocol_binding="HTTP+JSON",
                protocol_version="0.3",
                url=f"http://{host}:{port}/a2a/rest",
            ),
        ],
    )

    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=SampleAgentExecutor(),
        task_store=task_store,
        agent_card=agent_card,
    )

    rest_routes = create_rest_routes(
        request_handler=request_handler,
        path_prefix="/a2a/rest",
        enable_v0_3_compat=True,
    )
    jsonrpc_routes = create_jsonrpc_routes(
        request_handler=request_handler,
        rpc_url="/a2a/jsonrpc",
        enable_v0_3_compat=True,
    )
    agent_card_routes = create_agent_card_routes(
        agent_card=agent_card,
    )

    app = FastAPI(title="Healthcare Triage Agent")
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=agent_card_routes,
        jsonrpc_routes=jsonrpc_routes,
        rest_routes=rest_routes,
    )

    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)

    logger.info("Starting Healthcare Agent on http://%s:%s", host, port)
    logger.info(
        "Agent Card available at http://%s:%s/.well-known/agent-card.json",
        host,
        port,
    )
    logger.info("JSON-RPC endpoint at http://%s:%s/a2a/jsonrpc", host, port)
    logger.info("REST endpoint at http://%s:%s/a2a/rest", host, port)

    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2A Healthcare Agent")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=41241, help="Port to bind to")
    args = parser.parse_args()

    serve(host=args.host, port=args.port)
