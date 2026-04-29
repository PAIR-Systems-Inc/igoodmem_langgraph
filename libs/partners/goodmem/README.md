# langgraph-goodmem

An integration package connecting [GoodMem](https://goodmem.ai) and [LangGraph](https://github.com/langchain-ai/langgraph).

GoodMem is memory layer for AI agents with support for semantic storage, retrieval, and summarization. This package exposes GoodMem operations as LangGraph tools that can be used with any LangGraph agent.

## Installation

```bash
pip install langgraph-goodmem
```

## Tools

| Tool | Description |
|---|---|
| `GoodMemCreateSpace` | Create a new space or reuse an existing one |
| `GoodMemListSpaces` | List all spaces in your account |
| `GoodMemCreateMemory` | Store text or files as memories |
| `GoodMemRetrieveMemories` | Semantic similarity search across spaces |
| `GoodMemGetMemory` | Fetch a specific memory by ID |
| `GoodMemDeleteMemory` | Permanently delete a memory |
| `GoodMemListEmbedders` | List available embedder models |

## Quick start

```python
from langgraph_goodmem import GoodMemCreateSpace, GoodMemCreateMemory, GoodMemRetrieveMemories

# Create tools with your credentials
create_space = GoodMemCreateSpace(
    goodmem_base_url="http://localhost:8080",
    goodmem_api_key="your-api-key",
)

create_memory = GoodMemCreateMemory(
    goodmem_base_url="http://localhost:8080",
    goodmem_api_key="your-api-key",
)

retrieve = GoodMemRetrieveMemories(
    goodmem_base_url="http://localhost:8080",
    goodmem_api_key="your-api-key",
)

# Use with a LangGraph ReAct agent
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model="gpt-4o",
    tools=[create_space, create_memory, retrieve],
)
```

## Usage in a LangGraph graph

You can also use these tools as nodes in a custom LangGraph graph:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph_goodmem import (
    GoodMemCreateSpace,
    GoodMemCreateMemory,
    GoodMemRetrieveMemories,
)

# Initialize tools
tools = [
    GoodMemCreateSpace(
        goodmem_base_url="http://localhost:8080",
        goodmem_api_key="your-api-key",
    ),
    GoodMemCreateMemory(
        goodmem_base_url="http://localhost:8080",
        goodmem_api_key="your-api-key",
    ),
    GoodMemRetrieveMemories(
        goodmem_base_url="http://localhost:8080",
        goodmem_api_key="your-api-key",
    ),
]

# Create a ToolNode for use in your graph
tool_node = ToolNode(tools)
```

## Environment variables

| Variable | Description |
|---|---|
| `GOODMEM_BASE_URL` | Base URL of the GoodMem API server |
| `GOODMEM_API_KEY` | API key for authentication |
| `GOODMEM_VERIFY_SSL` | Set to `true` to verify SSL certificates (default: `true`) |
