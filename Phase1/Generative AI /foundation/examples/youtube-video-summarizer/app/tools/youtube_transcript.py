from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain.tools import tool


@tool
def get_youtube_transcript(video_id: str):
    """Fetches and formats the transcript of a YouTube video.

    Args:
        video_id: The 11-character YouTube video ID.

    Returns:
        str: The full transcript text, or an error message if retrieval fails.
    """
    try:
        # Fetch the raw transcript (returns a list of dictionaries with text, start, duration)
        raw_transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])

        # Format the transcript into a clean block of text
        formatter = TextFormatter()
        clean_text = formatter.format_transcript(raw_transcript)
        return clean_text
    except Exception as e:
        return f"Error retrieving transcript: {str(e)}"
