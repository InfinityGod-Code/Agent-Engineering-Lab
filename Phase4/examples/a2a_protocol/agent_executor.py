from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Part, Role
from pydantic import BaseModel


class SampleAgent(BaseModel):
    async def invoke(self, input_data: str) -> str:
        return f"Agent executed with input: {input_data}"


class SampleAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agent = SampleAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        result = await self.agent.invoke("Sample input")
        await event_queue.enqueue_event(
            Message(role=Role.ROLE_AGENT, parts=[Part(text=result)])
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise Exception("Cancel not supported")
