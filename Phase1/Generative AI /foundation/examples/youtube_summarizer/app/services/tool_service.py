from app.tools.get_transcript import get_transcript_tool


class ToolService:
    """
    A service class that provides access to various tools.
    """

    @staticmethod
    def get_summarize_tool():
        """
        Returns a Tool instance for summarizing text.
        """
        return [get_transcript_tool]
