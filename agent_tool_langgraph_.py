import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

assert os.environ["GOOGLE_API_KEY"]

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

@tool
def multiply_tool(a, b):
    """
    Multiply two numbers.
    Args:
        a: The first number
        b: The second number
    """
    return a * b

tools = [multiply_tool]

llm_with_tools = llm.bind_tools(tools)

messages = [
    HumanMessage(content="Calculate 999 * 1234"),
]

response = llm_with_tools.invoke(messages)

print(response)
