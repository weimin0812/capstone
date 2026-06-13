import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys


async def run_test():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["m4l1.py"]
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await  session.initialize()
            result = await session.call_tool("get_restaurant_info", arguments={"restaurant_name":"Iron"})


if __name__ == '__main__':
    asyncio.run(run_test())
