#!/usr/bin/env python3
"""
Full integration test for Rhetor MCP tools with live components.
Tests the complete Phase 3 implementation.
"""

import asyncio
import httpx
import json
import sys

async def test_full_integration():
    """Test full MCP integration with live components."""
    base_url = "http://localhost:8003"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("\n" + "="*60)
        print("Rhetor MCP Integration Test - Phase 3")
        print("="*60)
        
        # 1. Test MCP health
        print("\n1. Testing MCP Health")
        try:
            response = await client.get(f"{base_url}/api/mcp/v2/health")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 2. Test capabilities
        print("\n2. Testing Capabilities")
        try:
            response = await client.get(f"{base_url}/api/mcp/v2/capabilities")
            if response.status_code == 200:
                data = response.json()
                print(f"   Found {data['count']} capabilities:")
                for cap in data.get('capabilities', []):
                    print(f"   - {cap['name']}: {cap['description']}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 3. Test tools listing
        print("\n3. Testing Tools Listing")
        try:
            response = await client.get(f"{base_url}/api/mcp/v2/tools")
            if response.status_code == 200:
                data = response.json()
                print(f"   Found {data['count']} tools")
                # Show first 5 tools
                for tool in data.get('tools', [])[:5]:
                    print(f"   - {tool['name']}: {tool['description'][:60]}...")
                if data['count'] > 5:
                    print(f"   ... and {data['count'] - 5} more")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 4. Test AI Specialists listing (with live integration)
        print("\n4. Testing AI Specialists (Live Integration)")
        try:
            response = await client.post(
                f"{base_url}/api/mcp/v2/process",
                json={
                    "tool_name": "ListAISpecialists",
                    "arguments": {}
                }
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    specialists = data['result'].get('specialists', [])
                    print(f"   Found {len(specialists)} specialists:")
                    for spec in specialists:
                        status = "🟢 Active" if spec['active'] else "⭕ Inactive"
                        print(f"   - {spec['name']} ({spec['id']}): {status}")
                        print(f"     Role: {spec['description'][:50]}...")
                else:
                    print(f"   Error: {data.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 5. Test activating a specialist
        print("\n5. Testing Specialist Activation")
        try:
            response = await client.post(
                f"{base_url}/api/mcp/v2/process",
                json={
                    "tool_name": "ActivateAISpecialist",
                    "arguments": {
                        "specialist_id": "code-analyst"
                    }
                }
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    print(f"   Specialist activated: {data['result'].get('specialist_id')}")
                    print(f"   Status: {data['result'].get('status')}")
                else:
                    print(f"   Error: {data.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 6. Test sending a message to specialist
        print("\n6. Testing Message to Specialist")
        try:
            response = await client.post(
                f"{base_url}/api/mcp/v2/process",
                json={
                    "tool_name": "SendMessageToSpecialist",
                    "arguments": {
                        "specialist_id": "code-analyst",
                        "message": "Analyze the MCP integration code in Rhetor and suggest improvements."
                    }
                }
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    result = data['result']
                    print(f"   Message sent successfully!")
                    print(f"   Specialist: {result.get('specialist_id')}")
                    print(f"   Response preview: {result.get('response', '')[:100]}...")
                else:
                    print(f"   Error: {data.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 7. Test LLM model listing
        print("\n7. Testing LLM Model Management")
        try:
            response = await client.post(
                f"{base_url}/api/mcp/v2/process",
                json={
                    "tool_name": "GetAvailableModels",
                    "arguments": {}
                }
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    providers = data['result'].get('providers', {})
                    print(f"   Found {len(providers)} providers:")
                    for provider, info in providers.items():
                        models = info.get('models', [])
                        print(f"   - {provider}: {len(models)} models")
                        for model in models[:2]:  # Show first 2 models
                            print(f"     • {model['id']}")
                else:
                    print(f"   Error: {data.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 8. Test workflow execution
        print("\n8. Testing LLM Workflow Execution")
        try:
            response = await client.post(
                f"{base_url}/api/mcp/v2/execute-llm-workflow",
                json={
                    "workflow_name": "prompt_optimization",
                    "parameters": {
                        "base_prompt": "Explain the concept of recursion",
                        "task_context": {"audience": "beginners", "format": "simple"},
                        "optimization_goals": ["clarity", "engagement"]
                    }
                }
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"   Workflow executed successfully!")
                    summary = data['result'].get('workflow_summary', {})
                    print(f"   Optimization confidence: {summary.get('optimization_confidence')}")
                    print(f"   Improvements made: {summary.get('improvements_made', 0)}")
                else:
                    print(f"   Error: {data.get('detail', 'Unknown error')}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\n" + "="*60)
        print("Integration Test Complete")
        print("="*60)

if __name__ == "__main__":
    asyncio.run(test_full_integration())