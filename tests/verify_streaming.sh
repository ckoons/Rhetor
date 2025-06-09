#!/bin/bash
# Verify Rhetor streaming functionality

echo "🚀 Rhetor Streaming Verification Script"
echo "======================================"

# Check if Rhetor is running
echo -n "Checking if Rhetor is running... "
if curl -s http://localhost:8003/api/health > /dev/null 2>&1; then
    echo "✅ Rhetor is running"
else
    echo "❌ Rhetor is not running"
    echo "Please start Rhetor with: tekton-launch -c rhetor"
    exit 1
fi

# Test basic SSE endpoint
echo -e "\n📡 Testing basic SSE streaming endpoint..."
echo "Command: curl -N http://localhost:8003/api/mcp/v2/stream/test"
timeout 6 curl -N http://localhost:8003/api/mcp/v2/stream/test 2>/dev/null | head -20
echo -e "\n✅ Basic SSE test complete"

# Test MCP tools endpoint
echo -e "\n🔧 Checking available MCP tools..."
TOOLS_COUNT=$(curl -s http://localhost:8003/api/mcp/v2/tools | jq '. | length')
echo "Found $TOOLS_COUNT MCP tools"

# Check for streaming tools
echo -e "\n🌊 Checking for streaming-enabled tools..."
curl -s http://localhost:8003/api/mcp/v2/tools | jq -r '.tools[] | select(.name | contains("Stream")) | .name' 2>/dev/null || echo "Unable to parse tools list"

echo -e "\n✅ Streaming verification complete!"
echo -e "\nNext steps:"
echo "1. Run the Python test client: python /Rhetor/tests/test_streaming.py"
echo "2. Open the HTML client: open /Rhetor/examples/streaming_client.html"
echo "3. Check the streaming guide: cat /Rhetor/docs/STREAMING_GUIDE.md"