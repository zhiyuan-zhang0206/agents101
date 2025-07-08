import os
import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

assert os.environ["GOOGLE_API_KEY"]

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

def multiply_tool(a, b):
    return a * b

SYSTEM_PROMPT = """你必须用JSON格式回复，包含以下结构：
{
  "thoughts": "你的思考过程",
  "tool_call": {
    "name": "工具名称",
    "parameters": {"a": 数字1, "b": 数字2}
  }
}

可用工具：
- multiply: 乘法计算。Args: a, b

如果不需要工具，tool_call设为null。"""


def execute_tool(tool_name, params):
    if tool_name == "multiply":
        return multiply_tool(params["a"], params["b"])
    return None


messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content="Calculate 999 * 1234"),
]
response = model.invoke(messages)

print(response)

messages.append(AIMessage(content=response.content))

parsed_response = json.loads(str(response.content))
thoughts, tool_call = parsed_response["thoughts"], parsed_response["tool_call"]

tool_result = execute_tool(tool_call["name"], tool_call["parameters"])

messages.append(HumanMessage(content=f"Tool result: {tool_result}"))

response = model.invoke(messages)

print(response.content)  # I have used the tool to calculate that 999 * 1234 = ...