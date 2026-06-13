# curl -O https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/q5CWrWtux9-NmnkiWOrQBw/augmented-user-review.json
#
# curl -O https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/taf6n5kHS3H2ZLRf9tihFw/California-Culinary-Map.txt
#
# curl -O https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ql0xua4ET4XmQCOpR1lrNg/server.py
#
# curl -O https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/clYj6XUC9urkUzWtiDH0Cw/structured-restaurant-data.json

# Libraries to create our MCP host application
import os
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
from fastmcp.client import Client, PythonStdioTransport
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# 加载环境变量
load_dotenv()

# Configuration
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")

SYSTEM_PROMPT = """
You are a helpful California restaurant guide. You have access to tools to look up restaurant information,
find restaurants by vibe/atmosphere, and retrieve customer reviews.

Follow these rules:
1. Always use the available tools to get real restaurant data before answering the user.
2. Use get_restaurant_info when the user asks about a specific restaurant name.
3. Use recommend_by_vibe when the user describes a mood, atmosphere or dining style.
4. Use get_review when the user wants customer reviews for a restaurant.
5. Summarize the tool results clearly and naturally for the user.
6. Do not make up information; only use data returned from tools.

Answer in a friendly, conversational tone.
"""

def make_model():
    return ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0.7,
    )

# MCP Host — ReAct Agent Loop
async def chat_with_agent(user_message: str, history: list) -> str:
    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)

    async with Client(transport) as client:
        mcp_tools = await client.list_tools()

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in mcp_tools
        ]

        model = make_model().bind_tools(openai_tools)

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_message))

        for _ in range(10):
            response = await model.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                raw = response.content
                if isinstance(raw, list):
                    return " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in raw
                    )
                return str(raw)

            for tool_call in response.tool_calls:
                result = await client.call_tool(tool_call["name"], tool_call["args"])
                tool_output = " ".join(
                    item.text if hasattr(item, "text") else str(item)
                    for item in result.content
                ) if result.content else "(no result)"
                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

        return "I wasn't able to complete that request. Please try again."

# Gradio Event Handler
async def handle_chat(user_message, history):
    if history is None:
        history = []
    if not user_message or not user_message.strip():
        yield history
        return

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "Thinking..."},
    ]
    yield history

    response_text = await chat_with_agent(user_message, history[:-2])
    history[-1] = {"role": "assistant", "content": response_text}
    yield history

# Gradio Interface
with gr.Blocks(title="Connoisseur Companion") as demo:
    gr.Markdown("# Connoisseur Companion\nYour AI guide to California's restaurant scene. Ask me about restaurants by name, cuisine, or vibe!")

    # 已移除不兼容的 type="messages"
    chatbot = gr.Chatbot(height=500)
    msg_input = gr.Textbox(
        label="Ask about restaurants",
        placeholder='e.g., "Find me a moody spot in DTLA" or "Tell me about Sakura Garden"',
    )

    with gr.Row():
        btn1 = gr.Button("Find moody restaurants", size="sm")
        btn2 = gr.Button("Tell me about Iron & Embers", size="sm")
        btn3 = gr.Button("Zen dining in Little Tokyo?", size="sm")

    msg_input.submit(handle_chat, [msg_input, chatbot], [chatbot])
    msg_input.submit(lambda: "", None, msg_input)

    btn1.click(handle_chat, [gr.State("Find me some moody restaurants"), chatbot], [chatbot])
    btn2.click(handle_chat, [gr.State("Tell me about Iron & Embers"), chatbot])
    btn3.click(handle_chat, [gr.State("What's a zen dining experience in Little Tokyo?"), chatbot])

# Launch the App
if __name__ == "__main__":
    print("Starting Connoisseur Companion with DeepSeek...")
    demo.launch(
        share=True,
        theme=gr.themes.Soft(),
    )