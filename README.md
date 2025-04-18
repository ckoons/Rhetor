# Rhetor

## Overview

Rhetor is the LLM management system for the Tekton ecosystem. It handles prompt engineering, LLM selection, and context management for all AI interactions.

## Key Features

- LLM selection and routing
- Prompt template management
- Context management for LLM interactions
- Response processing and formatting
- Multi-model support

## Quick Start

```bash
# Register with Hermes
python -m Rhetor/register_with_hermes.py

# Start with Tekton
./scripts/tekton_launch --components rhetor
```

## Documentation

For detailed documentation, see the following resources in the MetaData directory:

- [Component Summaries](../MetaData/ComponentSummaries.md) - Overview of all Tekton components
- [Tekton Architecture](../MetaData/TektonArchitecture.md) - Overall system architecture
- [Component Integration](../MetaData/ComponentIntegration.md) - How components interact
- [CLI Operations](../MetaData/CLI_Operations.md) - Command-line operations