#!/usr/bin/env python3
"""Simple test to debug MCP process endpoint."""

import asyncio
import aiohttp
import json

async def test_process_endpoint():
    """Test the MCP process endpoint directly."""
    base_url = "http://localhost:8003/api/mcp/v2"
    
    async with aiohttp.ClientSession() as session:
        # First check what tools are available
        print("1. Checking available tools...")
        async with session.get(f"{base_url}/tools") as resp:
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                data = await resp.json()
                print(f"   Response: {json.dumps(data, indent=2)}")
            else:
                print(f"   Error: {await resp.text()}")
        
        print("\n2. Testing process endpoint with minimal payload...")
        payload = {
            "tool_name": "ListAISpecialists",
            "arguments": {}
        }
        async with session.post(f"{base_url}/process", json=payload) as resp:
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                data = await resp.json()
                print(f"   Response type: {type(data)}")
                print(f"   Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                print(f"   Full response: {json.dumps(data, indent=2)}")
            else:
                print(f"   Error: {await resp.text()}")
        
        print("\n3. Testing with context...")
        payload = {
            "tool_name": "ListAISpecialists",
            "arguments": {},
            "context": {}
        }
        async with session.post(f"{base_url}/process", json=payload) as resp:
            print(f"   Status: {resp.status}")
            if resp.status == 200:
                data = await resp.json()
                print(f"   Response: {json.dumps(data, indent=2) if data else 'None'}")

if __name__ == "__main__":
    asyncio.run(test_process_endpoint())