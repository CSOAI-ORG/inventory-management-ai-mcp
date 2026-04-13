# Inventory Management AI MCP Server
**By MEOK AI Labs** | [meok.ai](https://meok.ai)

Stock control toolkit: reorder points, demand forecasting, SKU optimization, warehouse layout, and shrinkage detection.

## Tools

| Tool | Description |
|------|-------------|
| `reorder_point` | Calculate reorder point, safety stock, and economic order quantity |
| `demand_forecast` | Forecast demand with moving average, exponential smoothing, or trend |
| `sku_optimizer` | ABC/XYZ classification with inventory strategy recommendations |
| `warehouse_layout` | Plan warehouse zones optimized for picking efficiency |
| `shrinkage_detector` | Detect inventory shrinkage anomalies with category analysis |

## Installation

```bash
pip install mcp
```

## Usage

### Run the server

```bash
python server.py
```

### Claude Desktop config

```json
{
  "mcpServers": {
    "inventory-management": {
      "command": "python",
      "args": ["/path/to/inventory-management-ai-mcp/server.py"]
    }
  }
}
```

## Pricing

| Tier | Limit | Price |
|------|-------|-------|
| Free | 30 calls/day | $0 |
| Pro | Unlimited + premium features | $9/mo |
| Enterprise | Custom + SLA + support | Contact us |

## License

MIT
