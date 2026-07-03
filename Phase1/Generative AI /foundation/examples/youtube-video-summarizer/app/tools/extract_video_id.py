from langchain.tools import tool
import re


@tool
def extract_video_id(url : str) :
    
    """
    Extracts the 11-character YouTube video ID from a URL.

    Args : YouTube url containing video id.

    Returns : 
        str : Extracted video id or the error message if parsing fails.
    """
    # Regex pattern to match video IDs
    pattern = r'(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern=pattern,string=url)
    return match.group(1) if match else "Error: Invalid YouTube URL"