#!/usr/bin/env python3
"""Simple test to debug the process endpoint."""

import asyncio
import httpx
import json

async def test_process():
    """Test the process endpoint directly."""
    base_url = "http://localhost:8003"
    
    async with httpx.AsyncClient() as client:
        # First check if the server is running
        try:
            response = await client.get(f"{base_url}/health")
            print(f"Health check: {response.status_code}")
        except Exception as e:
            print(f"Server not running: {e}")
            return
        
        # Test the process endpoint
        print("\nTesting /process endpoint...")
        request_data = {
            "tool_name": "ListAISpecialists",
            "arguments": {}
        }
        
        try:
            response = await client.post(
                f"{base_url}/api/mcp/v2/process",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
            else:
                print(f"Response text: {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_process())