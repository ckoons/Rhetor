#!/usr/bin/env python3
"""Debug reimport of the module."""

import sys
import importlib

# Remove any cached versions
modules_to_remove = []
for module_name in sys.modules:
    if 'tekton.mcp.fastmcp' in module_name:
        modules_to_remove.append(module_name)

for module_name in modules_to_remove:
    del sys.modules[module_name]
    print(f"Removed cached module: {module_name}")

# Now reimport
from tekton.mcp.fastmcp.utils.endpoints import add_mcp_endpoints

# Check the source
import inspect
source = inspect.getsource(add_mcp_endpoints)

# Find the process_request function
lines = source.split('\n')
in_process_func = False
for i, line in enumerate(lines):
    if '@router.post("/process")' in line:
        in_process_func = True
        print(f"\nFound process endpoint at line {i}:")
    
    if in_process_func:
        print(f"  {i}: {line}")
        
        if 'return router' in line and not line.strip().startswith('#'):
            break