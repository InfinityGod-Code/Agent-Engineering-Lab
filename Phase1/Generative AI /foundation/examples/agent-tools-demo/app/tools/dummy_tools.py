from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location"""
    return f"The weather in {location} is sunny and 72°F."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression"""
    return f"Result: {eval(expression)}"


dummy_tools = [get_weather, calculator]
