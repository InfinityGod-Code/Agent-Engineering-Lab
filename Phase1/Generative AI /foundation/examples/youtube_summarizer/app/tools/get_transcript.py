from langchain.tools import tool


@tool
def get_transcript_tool():
    """
    Returns a Tool instance for fetching YouTube video transcripts.
    """
    return {
        "name": "get_transcript",
        "description": "Fetches the transcript of a YouTube video given its URL.",
    }
