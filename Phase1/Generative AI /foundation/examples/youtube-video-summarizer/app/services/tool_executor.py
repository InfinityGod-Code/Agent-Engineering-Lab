from langchain.messages import ToolMessage
from app.tools.extract_video_id import extract_video_id
from app.tools.youtube_transcript import get_youtube_transcript


class ToolExecutor:
    @classmethod
    def execute(cls, tool_call):
        try:
            tool_mapper = {
                "extract_video_id": extract_video_id,
                "get_youtube_transcript": get_youtube_transcript,
            }
            tool_content = tool_mapper[tool_call["name"]].invoke(tool_call["args"])
            return ToolMessage(
                content=tool_content,
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        except Exception as e:
            return ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )

    @classmethod
    def get_all_tools(cls):
        return [extract_video_id, get_youtube_transcript]
