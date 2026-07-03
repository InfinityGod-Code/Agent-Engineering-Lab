import asyncio
from typing import Any, AsyncGenerator

from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from app.services.tool_executor import ToolExecutor


class YTChain:
    def __init__(self, tools, model):
        self.tools = tools
        self.model = model
        self.llm_with_tools = self.model.get_model().bind_tools(self.tools)

    def _build_chain(self):
        return (
            RunnablePassthrough.assign(messages=lambda x: [HumanMessage(x["query"])])
            | RunnablePassthrough.assign(
                ai_response=lambda x: self.llm_with_tools.invoke(x["messages"])
            )
            | RunnablePassthrough.assign(
                tool_response=lambda x: [
                    ToolExecutor.execute(tool_call=tool)
                    for tool in x["ai_response"].tool_calls
                ]
            )
            | RunnablePassthrough.assign(
                messages=lambda x: (
                    x["messages"] + [x["ai_response"]] + x["tool_response"]
                )
            )
            | RunnablePassthrough.assign(
                ai_response2=lambda x: self.llm_with_tools.invoke(x["messages"])
            )
            | RunnablePassthrough.assign(
                tool_response2=lambda x: [
                    ToolExecutor.execute(tool_call=tool)
                    for tool in x["ai_response2"].tool_calls
                ]
            )
            | RunnablePassthrough.assign(
                messages=lambda x: (
                    x["messages"] + [x["ai_response2"]] + x["tool_response2"]
                )
            )
            | RunnablePassthrough.assign(
                summary=lambda x: self.llm_with_tools.invoke(x["messages"]).content
            )
            | RunnableLambda(lambda x: x["summary"])
        )

    def run(self, url: str) -> Any:
        return self._build_chain().invoke(
            {"query": f"Please summarize the YouTube video at this URL: {url}"}
        )

    async def stream_run(self, url: str) -> AsyncGenerator[dict, None]:
        base_model = self.model.get_model()
        messages = [
            HumanMessage(
                content=f"Please summarize the YouTube video at this URL: {url}"
            )
        ]

        yield {"type": "status", "data": {"message": "Analyzing video URL..."}}

        resp1 = await asyncio.to_thread(self.llm_with_tools.invoke, messages)
        tool_msgs1 = []
        for tc in resp1.tool_calls:
            yield {
                "type": "tool_start",
                "data": {"name": tc["name"], "args": tc["args"]},
            }
            result = await asyncio.to_thread(ToolExecutor.execute, tc)
            tool_msgs1.append(result)
            yield {
                "type": "tool_end",
                "data": {"name": tc["name"], "result": result.content},
            }

        messages.extend([resp1] + tool_msgs1)

        yield {"type": "status", "data": {"message": "Fetching transcript..."}}

        resp2 = await asyncio.to_thread(self.llm_with_tools.invoke, messages)
        tool_msgs2 = []
        for tc in resp2.tool_calls:
            yield {
                "type": "tool_start",
                "data": {"name": tc["name"], "args": tc["args"]},
            }
            result = await asyncio.to_thread(ToolExecutor.execute, tc)
            tool_msgs2.append(result)
            yield {
                "type": "tool_end",
                "data": {"name": tc["name"], "result": result.content},
            }

        messages.extend([resp2] + tool_msgs2)

        yield {"type": "status", "data": {"message": "Generating summary..."}}

        async for chunk in base_model.astream(messages):
            if isinstance(chunk, AIMessageChunk):
                text = ""
                if isinstance(chunk.content, str):
                    text = chunk.content
                elif isinstance(chunk.content, list):
                    for block in chunk.content:
                        if isinstance(block, dict) and block.get("text"):
                            text += block["text"]
                if text:
                    yield {"type": "token", "data": text}

        yield {"type": "done", "data": {}}
