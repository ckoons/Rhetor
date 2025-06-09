#!/usr/bin/env python3
"""Debug which endpoints.py file is being imported."""

import sys
sys.path.insert(0, '/Users/cskoons/projects/github/Tekton')

try:
    from tekton.mcp.fastmcp.utils.endpoints import add_mcp_endpoints
    print("Imported from tekton.mcp.fastmcp.utils.endpoints")
    print(f"Module path: {add_mcp_endpoints.__module__}")
    
    # Try to get the file path
    import inspect
    print(f"File path: {inspect.getfile(add_mcp_endpoints)}")
    
    # Check the source code
    source = inspect.getsource(add_mcp_endpoints)
    # Find the process_request function
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if 'process_request_func(request.dict(), None)' in line:
            print(f"\nFound at line {i}: {line.strip()}")
            # Show context
            for j in range(max(0, i-2), min(len(lines), i+3)):
                print(f"  {j}: {lines[j]}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()