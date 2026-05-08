<div align="center">

# Inventory Management Ai MCP

**MCP server for inventory management ai mcp operations**

[![PyPI](https://img.shields.io/pypi/v/meok-inventory-management-ai-mcp)](https://pypi.org/project/meok-inventory-management-ai-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MEOK AI Labs](https://img.shields.io/badge/MEOK_AI_Labs-MCP_Server-purple)](https://meok.ai)

</div>

## Overview

Inventory Management Ai MCP provides AI-powered tools via the Model Context Protocol (MCP).

## Tools

| Tool | Description |
|------|-------------|
| `reorder_point` | Calculate optimal reorder point, safety stock, and economic order quantity |
| `demand_forecast` | Forecast future demand from historical sales data with confidence intervals. |
| `sku_optimizer` | Classify SKUs using ABC/XYZ analysis and recommend inventory strategies. |
| `warehouse_layout` | Plan warehouse zone layout optimized for picking efficiency. |
| `shrinkage_detector` | Detect inventory shrinkage by comparing expected vs actual quantities. |

## Installation

```bash
pip install meok-inventory-management-ai-mcp
```

## Usage with Claude Desktop

Add to your Claude Desktop MCP config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "inventory-management-ai-mcp": {
      "command": "python",
      "args": ["-m", "meok_inventory_management_ai_mcp.server"]
    }
  }
}
```

## Usage with FastMCP

```python
from mcp.server.fastmcp import FastMCP

# This server exposes 5 tool(s) via MCP
# See server.py for full implementation
```

## License

MIT © [MEOK AI Labs](https://meok.ai)
