#!/usr/bin/env python3
"""Test script to verify MCP tools are properly exposed."""

import asyncio
import httpx
import json

async def test_mcp_tools():
    """Test that MCP tools are properly registered and exposed."""
    base_url = "http://localhost:8003"
    
    async with httpx.AsyncClient() as client:
        # Test debug endpoint
        print("\n=== Testing Debug Endpoint ===")
        try:
            response = await client.get(f"{base_url}/api/mcp/v2/debug/tools")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Debug data: {json.dumps(data, indent=2)}")
        except Exception as e:
            print(f"Debug endpoint error: {e}")
        
        # Test tools endpoint
        print("\n=== Testing Tools Endpoint ===")
        try:
            response = await client.get(f"{base_url}/api/mcp/v2/tools")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Tools count: {data.get('count', 0)}")
                if data.get('tools'):
                    print("\nRegistered tools:")
                    for tool in data['tools']:
                        print(f"  - {tool['name']}: {tool['description']}")
                else:
                    print("No tools returned!")
                    print(f"Full response: {json.dumps(data, indent=2)}")
        except Exception as e:
            print(f"Tools endpoint error: {e}")
        
        # Test capabilities endpoint
        print("\n=== Testing Capabilities Endpoint ===")
        try:
            response = await client.get(f"{base_url}/api/mcp/v2/capabilities")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Capabilities count: {data.get('count', 0)}")
                if data.get('capabilities'):
                    print("\nRegistered capabilities:")
                    for cap in data['capabilities']:
                        print(f"  - {cap['name']}: {cap['description']}")
        except Exception as e:
            print(f"Capabilities endpoint error: {e}")
        
        # Test a specific tool
        print("\n=== Testing Tool Execution ===")
        try:
            request_data = {
                "tool_name": "ListAISpecialists",
                "arguments": {}
            }
            response = await client.post(
                f"{base_url}/api/mcp/v2/process",
                json=request_data
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Execution result: {json.dumps(data, indent=2)}")
        except Exception as e:
            print(f"Tool execution error: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcp_tools())