"""MCP Client configuration"""
import asyncio
from fastmcp import Client

# MCP Client instance - used by server.py
client = Client("http://localhost:8000/mcp")

# Test function - run this file directly to test MCP connection
async def main():
    async with client:
        await client.ping()
        print("Server is reachable")
        print(f"Connected: {client.is_connected()}")
        
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")
        

if __name__ == "__main__":
    asyncio.run(main())