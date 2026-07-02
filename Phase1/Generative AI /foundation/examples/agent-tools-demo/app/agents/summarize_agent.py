from langchain import create_agent
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from app.services.tool_service import ToolService
from rich.console import Console


class SummarizeAgent:
    def __init__(self, model):
        self.model = model
        self.agent = create_agent(
            llm=self.model.get_llm(),
            tools=ToolService.get_summarize_tool(),
            agent_type="zero-shot-react-description",
            verbose=True,
        )

    def stream_events(self, input_text):
        stream =  self.agent.stream_events(
            {"messages": [{"role": "user", "content": input_text}]},
            version="v3",
        )

        for chunk in stream:
            if chunk["type"] == "messages":
                token, metadata = chunk["data"]
                if isinstance(token, AIMessageChunk):
                    self._render_message_chunk(token)
            elif chunk["type"] == "updates":
                for source, update in chunk["data"].items():
                    if source in ("model", "tools"): 
                        self._render_completed_message(update["messages"][-1])
    

    def _render_message_chunk(token: AIMessageChunk) -> None:
            if token.text:
                Console().print(token.text, end="|")
            if token.tool_call_chunks:
                Console().print(token.tool_call_chunks, style="bold white on green")
    # N.B. all content is available through token.content_blocks


    def _render_completed_message(message: AnyMessage) -> None:

        if isinstance(message, AIMessage) and message.tool_calls:
            Console().print(f"Tool calls: {message.tool_calls}", style="bold white on green")
        if isinstance(message, ToolMessage):
            Console().print(f"Tool response: {message.content_blocks}", style="bold white on blue")
        