import asyncio
import json
import sys
from pathlib import Path
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    Root,
    TextContent,
    CreateMessageResult,
    CreateMessageRequestParams,
)

# Configuration
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")
PROJECT_DIR = Path(__file__).parent.resolve()

# StdioServerParameters: 使用当前环境的 Python 解释器启动服务端
server_params = StdioServerParameters(
    command=sys.executable,
    args=[SERVER_SCRIPT],
)

# Anthropic client used to fulfill sampling requests from the server
anthropic_client = Anthropic()


# SAMPLING — Handle LLM requests delegated from the server
async def handle_sampling(params: CreateMessageRequestParams) -> CreateMessageResult:
    """Run a Claude LLM call on behalf of the server and return the result."""
    # Extract the prompt text from the first sampling message
    prompt = params.messages[0].content.text

    print(f"\n[Sampling] Server requested LLM task:")
    print(f"  Prompt preview: {prompt[:150]}...")

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=params.maxTokens or 200,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.content[0].text
    print(f"  LLM Response: {response_text[:100]}...")

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=response_text),
        model="claude-sonnet-4-20250514",
    )


# ROOTS — Tell the server which directories it can access
def list_roots() -> list[Root]:
    """Limit the server's file access to this project directory."""
    return [Root(uri=f"file://{PROJECT_DIR}", name=PROJECT_DIR.name)]


async def verify_connection():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
                read,
                write,
                sampling_callback=handle_sampling,
                list_roots_callback=list_roots,
        ) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]
            resources_result = await session.list_resources()
            roots = list_roots()


async def demo_get_restaurant_info():
    """Demo: Look up a restaurant by name."""
    print("\n" + "-" * 60)
    print("Demo: get_restaurant_info('Iron & Embers')")
    print("-" * 60)

    data = await call_tool("get_restaurant_info", {"restaurant_name": "Iron & Embers"})
    print(json.dumps(data, indent=2))


async def call_tool(tool_name: str, arguments: dict) -> dict:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
                read,
                write,
                sampling_callback=handle_sampling,
                list_roots_callback=list_roots,
        ) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return json.loads(result.content[0].text)


async def demo_recommend_by_vibe():
    """Demo: Find restaurants by vibe keyword."""
    print("\n" + "-" * 60)
    print("Demo: recommend_by_vibe('moody')")
    print("-" * 60)

    data = await call_tool("recommend_by_vibe", {"vibe": "moody"})
    print(f"Vibe: {data['vibe_searched']}")
    print(f"Structured matches: {len(data['structured_matches'])}")
    for match in data["structured_matches"]:
        print(f"  - {match['name']} ({match['cuisine']}) - {match['rating']}/5")
    print(f"Raw text excerpts: {len(data['raw_text_excerpts'])}")


async def demo_get_review():
    """Demo: Retrieve a restaurant review."""
    print("\n" + "-" * 60)
    print("Demo: get_review('Iron & Embers')")
    print("-" * 60)

    data = await call_tool("get_review", {"restaurant_name": "Iron & Embers"})
    print(json.dumps(data, indent=2))


async def main():
    """Run all demos sequentially."""
    await verify_connection()
    await demo_get_restaurant_info()
    await demo_recommend_by_vibe()
    await demo_get_review()


if __name__ == '__main__':
    asyncio.run(main())
