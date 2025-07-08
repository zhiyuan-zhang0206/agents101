import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

assert os.environ["GOOGLE_API_KEY"]

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

messages = []

while True:
    user_input = input("You: ")
    messages.append(HumanMessage(content=user_input))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print(f"AI: {response.content}")