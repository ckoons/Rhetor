#!/usr/bin/env python3
"""Debug sys.path to see what's being imported."""

import sys
print("Python sys.path:")
for i, path in enumerate(sys.path):
    if 'tekton' in path.lower():
        print(f"{i}: {path} <-- TEKTON PATH")
    else:
        print(f"{i}: {path}")

# Try importing and see which version we get
try:
    import tekton
    print(f"\ntekton module path: {tekton.__file__}")
except Exception as e:
    print(f"\nCouldn't import tekton: {e}")

# Check if tekton-core is on the path
print("\nChecking for tekton-core paths:")
for path in sys.path:
    if 'tekton-core' in path:
        print(f"  Found: {path}")