#!/usr/bin/env python3
"""
Inventory Management AI MCP Server
======================================
Stock control toolkit for AI agents: reorder point calculation, demand forecasting,
SKU optimization, warehouse layout planning, and shrinkage detection.

By MEOK AI Labs | https://meok.ai

Install: pip install mcp
Run:     python server.py
"""

import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
FREE_DAILY_LIMIT = 30
_usage: dict[str, list[datetime]] = defaultdict(list)


def _check_rate_limit(caller: str = "anonymous") -> Optional[str]:
    now = datetime.now()
    cutoff = now - timedelta(days=1)
    _usage[caller] = [t for t in _usage[caller] if t > cutoff]
    if len(_usage[caller]) >= FREE_DAILY_LIMIT:
        return f"Free tier limit reached ({FREE_DAILY_LIMIT}/day). Upgrade: https://mcpize.com/inventory-management-ai-mcp/pro"
    _usage[caller].append(now)
    return None


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------
def _reorder_point(avg_daily_demand: float, lead_time_days: int,
                   safety_stock_days: int, service_level: float,
                   demand_std_dev: float) -> dict:
    """Calculate reorder point and economic order quantity."""
    if avg_daily_demand <= 0:
        return {"error": "Average daily demand must be positive"}
    if lead_time_days <= 0:
        return {"error": "Lead time must be positive"}

    # Z-score approximation for service level
    z_scores = {0.90: 1.28, 0.95: 1.65, 0.97: 1.88, 0.99: 2.33, 0.999: 3.09}
    z = z_scores.get(service_level, 1.65)

    # Safety stock calculation
    if demand_std_dev > 0:
        safety_stock = z * demand_std_dev * math.sqrt(lead_time_days)
    else:
        safety_stock = avg_daily_demand * safety_stock_days

    # Reorder point
    rop = (avg_daily_demand * lead_time_days) + safety_stock

    # Economic Order Quantity (EOQ) - Wilson formula
    # Assuming order cost = $25, holding cost = 20% of unit cost ($10 default)
    order_cost = 25.0
    annual_demand = avg_daily_demand * 365
    holding_cost = 2.0  # per unit per year
    eoq = math.sqrt((2 * annual_demand * order_cost) / holding_cost)

    # Order frequency
    orders_per_year = annual_demand / eoq if eoq > 0 else 0
    days_between_orders = 365 / orders_per_year if orders_per_year > 0 else 0

    return {
        "reorder_point": round(rop, 1),
        "safety_stock": round(safety_stock, 1),
        "inputs": {
            "avg_daily_demand": avg_daily_demand,
            "lead_time_days": lead_time_days,
            "service_level": service_level,
            "z_score": z,
            "demand_std_dev": demand_std_dev,
        },
        "eoq": {
            "quantity": round(eoq, 0),
            "orders_per_year": round(orders_per_year, 1),
            "days_between_orders": round(days_between_orders, 1),
            "annual_order_cost": round(orders_per_year * order_cost, 2),
            "annual_holding_cost": round((eoq / 2) * holding_cost, 2),
        },
        "annual_demand": round(annual_demand, 0),
        "recommendation": (
            f"Reorder {round(eoq)} units when stock falls to {round(rop)} units. "
            f"Safety stock of {round(safety_stock)} units covers {service_level*100:.0f}% service level."
        ),
    }


def _demand_forecast(historical_data: list[float], periods_ahead: int,
                     method: str, seasonality_period: int) -> dict:
    """Forecast future demand from historical data."""
    if not historical_data or len(historical_data) < 3:
        return {"error": "Need at least 3 historical data points"}
    if periods_ahead <= 0 or periods_ahead > 52:
        return {"error": "Forecast 1-52 periods ahead"}

    n = len(historical_data)
    forecasts = []

    if method == "moving_average":
        window = min(3, n)
        last_avg = statistics.mean(historical_data[-window:])
        for i in range(periods_ahead):
            forecasts.append(round(last_avg, 2))

    elif method == "exponential_smoothing":
        alpha = 0.3
        smoothed = historical_data[0]
        for val in historical_data[1:]:
            smoothed = alpha * val + (1 - alpha) * smoothed
        for i in range(periods_ahead):
            forecasts.append(round(smoothed, 2))

    elif method == "linear_trend":
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(historical_data)
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(historical_data))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean
        for i in range(periods_ahead):
            forecasts.append(round(max(0, intercept + slope * (n + i)), 2))

    elif method == "seasonal":
        if seasonality_period <= 0 or seasonality_period > n:
            seasonality_period = min(12, n)
        seasonal_indices = []
        for i in range(seasonality_period):
            vals = [historical_data[j] for j in range(i, n, seasonality_period)]
            seasonal_indices.append(statistics.mean(vals))
        base = statistics.mean(historical_data[-seasonality_period:])
        idx_mean = statistics.mean(seasonal_indices) if seasonal_indices else 1
        for i in range(periods_ahead):
            idx = seasonal_indices[i % len(seasonal_indices)]
            factor = idx / idx_mean if idx_mean > 0 else 1
            forecasts.append(round(max(0, base * factor), 2))
    else:
        return {"error": f"Unknown method '{method}'. Use: moving_average, exponential_smoothing, linear_trend, seasonal"}

    # Confidence intervals (simplified)
    std = statistics.stdev(historical_data) if len(historical_data) > 1 else 0
    intervals = []
    for i, f in enumerate(forecasts):
        margin = 1.96 * std * math.sqrt(1 + (i + 1) / n)
        intervals.append({
            "period": i + 1,
            "forecast": f,
            "lower_95": round(max(0, f - margin), 2),
            "upper_95": round(f + margin, 2),
        })

    return {
        "method": method,
        "historical_periods": n,
        "forecast_periods": periods_ahead,
        "forecasts": intervals,
        "summary": {
            "historical_mean": round(statistics.mean(historical_data), 2),
            "historical_std": round(std, 2),
            "forecast_mean": round(statistics.mean(forecasts), 2),
            "trend": "increasing" if forecasts[-1] > forecasts[0] else "decreasing" if forecasts[-1] < forecasts[0] else "stable",
        },
    }


def _sku_optimizer(skus: list[dict]) -> dict:
    """Analyze SKUs with ABC/XYZ classification for inventory optimization."""
    if not skus:
        return {"error": "Provide SKUs as [{sku, revenue, quantity, cost}]"}

    total_revenue = sum(s.get("revenue", 0) for s in skus)
    if total_revenue <= 0:
        return {"error": "Total revenue must be positive"}

    # Sort by revenue descending
    sorted_skus = sorted(skus, key=lambda x: x.get("revenue", 0), reverse=True)

    classified = []
    cumulative_pct = 0

    for s in sorted_skus:
        rev = s.get("revenue", 0)
        qty = s.get("quantity", 0)
        cost = s.get("cost", 0)
        rev_pct = (rev / total_revenue) * 100
        cumulative_pct += rev_pct

        # ABC classification
        if cumulative_pct <= 80:
            abc = "A"
        elif cumulative_pct <= 95:
            abc = "B"
        else:
            abc = "C"

        # Demand variability (XYZ) - simplified
        cv = s.get("demand_cv", 0.3)
        if cv < 0.25:
            xyz = "X"  # Stable demand
        elif cv < 0.5:
            xyz = "Y"  # Variable demand
        else:
            xyz = "Z"  # Highly variable

        margin = ((rev - cost * qty) / max(rev, 1)) * 100 if qty > 0 else 0

        classified.append({
            "sku": s.get("sku", "unknown"),
            "revenue": rev,
            "revenue_pct": round(rev_pct, 2),
            "cumulative_pct": round(cumulative_pct, 2),
            "quantity": qty,
            "margin_pct": round(margin, 1),
            "abc_class": abc,
            "xyz_class": xyz,
            "combined_class": f"{abc}{xyz}",
            "strategy": {
                "AX": "Tight control, frequent review, JIT ordering",
                "AY": "Regular review, safety stock, demand monitoring",
                "AZ": "High safety stock, close monitoring, consider consignment",
                "BX": "Standard review, moderate safety stock",
                "BY": "Periodic review, standard safety stock",
                "BZ": "Safety stock buffer, review quarterly",
                "CX": "Minimal attention, bulk ordering",
                "CY": "Low priority, periodic check",
                "CZ": "Consider discontinuing or minimum stock only",
            }.get(f"{abc}{xyz}", "Standard management"),
        })

    summary = {
        "total_skus": len(skus),
        "total_revenue": round(total_revenue, 2),
        "a_count": sum(1 for s in classified if s["abc_class"] == "A"),
        "b_count": sum(1 for s in classified if s["abc_class"] == "B"),
        "c_count": sum(1 for s in classified if s["abc_class"] == "C"),
        "a_revenue_pct": round(sum(s["revenue_pct"] for s in classified if s["abc_class"] == "A"), 1),
    }

    return {
        "classification": classified[:50],
        "summary": summary,
        "pareto_check": f"{'Follows' if summary['a_count'] <= len(skus) * 0.3 else 'Deviates from'} 80/20 rule: {summary['a_count']} SKUs ({round(summary['a_count']/len(skus)*100, 1)}%) generate {summary['a_revenue_pct']}% of revenue",
    }


def _warehouse_layout(zones: list[dict], total_sqft: float,
                      picking_method: str) -> dict:
    """Plan warehouse zone layout based on product velocity and picking method."""
    if not zones:
        return {"error": "Provide zones as [{name, sku_count, daily_picks, storage_type}]"}
    if total_sqft <= 0:
        return {"error": "Total square footage must be positive"}

    methods = {
        "wave": {"description": "Batch orders into waves, pick multiple orders simultaneously", "efficiency": "high_volume"},
        "zone": {"description": "Pickers assigned to zones, orders passed between zones", "efficiency": "large_warehouse"},
        "batch": {"description": "Group similar orders, pick in batches", "efficiency": "medium_volume"},
        "discrete": {"description": "One order at a time, simple but slow", "efficiency": "low_volume"},
    }

    method_info = methods.get(picking_method, methods["zone"])
    total_picks = sum(z.get("daily_picks", 0) for z in zones)

    # Allocate space proportionally by pick frequency (fast movers get prime space)
    sorted_zones = sorted(zones, key=lambda z: z.get("daily_picks", 0), reverse=True)

    layout = []
    # Reserve areas
    dock_pct = 0.10
    staging_pct = 0.08
    aisle_pct = 0.15
    usable_sqft = total_sqft * (1 - dock_pct - staging_pct - aisle_pct)

    for i, zone in enumerate(sorted_zones):
        picks = zone.get("daily_picks", 0)
        pick_ratio = picks / max(total_picks, 1)
        # Fast movers get disproportionately more accessible space
        weight = pick_ratio * 1.3 if i < len(sorted_zones) * 0.3 else pick_ratio * 0.85
        allocated = usable_sqft * (weight / max(sum(
            (z.get("daily_picks", 0) / max(total_picks, 1)) * (1.3 if j < len(sorted_zones) * 0.3 else 0.85)
            for j, z in enumerate(sorted_zones)
        ), 0.01))

        storage = zone.get("storage_type", "shelf")
        height_multiplier = {"pallet_rack": 4, "shelf": 2, "bin": 1.5, "floor": 1}
        effective_sqft = allocated * height_multiplier.get(storage, 1)

        proximity = "Near dock" if i < len(sorted_zones) * 0.2 else "Mid-warehouse" if i < len(sorted_zones) * 0.6 else "Back of warehouse"

        layout.append({
            "zone": zone.get("name", f"Zone {i + 1}"),
            "sku_count": zone.get("sku_count", 0),
            "daily_picks": picks,
            "pick_pct": round(pick_ratio * 100, 1),
            "allocated_sqft": round(allocated, 0),
            "effective_sqft": round(effective_sqft, 0),
            "storage_type": storage,
            "proximity": proximity,
            "velocity_class": "A - Fast" if i < len(sorted_zones) * 0.2 else "B - Medium" if i < len(sorted_zones) * 0.6 else "C - Slow",
        })

    return {
        "total_sqft": total_sqft,
        "picking_method": picking_method,
        "method_description": method_info["description"],
        "space_allocation": {
            "dock_area": round(total_sqft * dock_pct),
            "staging_area": round(total_sqft * staging_pct),
            "aisle_space": round(total_sqft * aisle_pct),
            "storage_zones": round(usable_sqft),
        },
        "zones": layout,
        "optimization_tips": [
            "Place top 20% velocity SKUs nearest to shipping dock",
            "Use golden zone (waist-to-shoulder height) for fast movers",
            "Minimize travel distance for wave/batch picking",
            "Consider cross-docking for high-velocity pass-through items",
        ],
    }


def _shrinkage_detector(inventory_records: list[dict]) -> dict:
    """Analyze inventory records to detect potential shrinkage patterns."""
    if not inventory_records:
        return {"error": "Provide records as [{sku, expected_qty, actual_qty, value_per_unit, category}]"}

    total_expected_value = 0
    total_actual_value = 0
    anomalies = []

    category_shrinkage = defaultdict(lambda: {"expected": 0, "actual": 0, "count": 0})

    for record in inventory_records:
        sku = record.get("sku", "unknown")
        expected = record.get("expected_qty", 0)
        actual = record.get("actual_qty", 0)
        value = record.get("value_per_unit", 0)
        category = record.get("category", "general")

        expected_val = expected * value
        actual_val = actual * value
        total_expected_value += expected_val
        total_actual_value += actual_val

        variance = expected - actual
        variance_pct = (variance / max(expected, 1)) * 100

        category_shrinkage[category]["expected"] += expected_val
        category_shrinkage[category]["actual"] += actual_val
        category_shrinkage[category]["count"] += 1

        if variance_pct > 5 or variance > 10:
            severity = "HIGH" if variance_pct > 15 else "MEDIUM" if variance_pct > 10 else "LOW"
            anomalies.append({
                "sku": sku,
                "expected": expected,
                "actual": actual,
                "variance": variance,
                "variance_pct": round(variance_pct, 2),
                "value_lost": round(variance * value, 2),
                "severity": severity,
                "category": category,
            })

    total_shrinkage = total_expected_value - total_actual_value
    shrinkage_rate = (total_shrinkage / max(total_expected_value, 1)) * 100

    # Category breakdown
    cat_analysis = {}
    for cat, data in category_shrinkage.items():
        cat_loss = data["expected"] - data["actual"]
        cat_analysis[cat] = {
            "expected_value": round(data["expected"], 2),
            "actual_value": round(data["actual"], 2),
            "shrinkage_value": round(cat_loss, 2),
            "shrinkage_pct": round((cat_loss / max(data["expected"], 1)) * 100, 2),
            "sku_count": data["count"],
        }

    # Industry benchmark
    if shrinkage_rate < 1.0:
        assessment = "Below average - excellent inventory control"
    elif shrinkage_rate < 2.0:
        assessment = "Average - within industry norms (US retail avg: 1.4%)"
    elif shrinkage_rate < 3.5:
        assessment = "Above average - investigate causes"
    else:
        assessment = "Critical - immediate investigation required"

    anomalies.sort(key=lambda x: x["value_lost"], reverse=True)

    return {
        "total_expected_value": round(total_expected_value, 2),
        "total_actual_value": round(total_actual_value, 2),
        "total_shrinkage": round(total_shrinkage, 2),
        "shrinkage_rate_pct": round(shrinkage_rate, 2),
        "assessment": assessment,
        "anomaly_count": len(anomalies),
        "top_anomalies": anomalies[:10],
        "category_analysis": cat_analysis,
        "likely_causes": [
            "Administrative errors (data entry, miscounts)",
            "Vendor fraud (short shipments, mislabeling)",
            "Employee theft (especially high-value small items)",
            "Shoplifting (if retail)",
            "Damage / spoilage (especially perishables)",
        ],
        "recommendations": [
            "Conduct cycle counts on high-variance SKUs",
            "Review receiving procedures for accuracy",
            "Implement surveillance on high-shrinkage categories",
            "Train staff on proper inventory handling",
        ],
    }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Inventory Management AI MCP",
    instructions="Stock control toolkit: reorder points, demand forecasting, SKU optimization, warehouse layout, and shrinkage detection. By MEOK AI Labs.",
)


@mcp.tool()
def reorder_point(avg_daily_demand: float, lead_time_days: int = 7,
                  safety_stock_days: int = 3, service_level: float = 0.95,
                  demand_std_dev: float = 0.0) -> dict:
    """Calculate optimal reorder point, safety stock, and economic order quantity
    using statistical service level targeting.

    Args:
        avg_daily_demand: Average units sold per day
        lead_time_days: Supplier lead time in days
        safety_stock_days: Buffer days of safety stock
        service_level: Target service level (0.90, 0.95, 0.97, 0.99, 0.999)
        demand_std_dev: Standard deviation of daily demand (0 = use safety_stock_days instead)
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _reorder_point(avg_daily_demand, lead_time_days, safety_stock_days, service_level, demand_std_dev)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def demand_forecast(historical_data: list[float], periods_ahead: int = 6,
                    method: str = "exponential_smoothing",
                    seasonality_period: int = 12) -> dict:
    """Forecast future demand from historical sales data with confidence intervals.

    Args:
        historical_data: List of historical demand values (e.g. monthly sales)
        periods_ahead: Number of periods to forecast (1-52)
        method: Forecasting method (moving_average, exponential_smoothing, linear_trend, seasonal)
        seasonality_period: Period length for seasonal method (e.g. 12 for monthly data)
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _demand_forecast(historical_data, periods_ahead, method, seasonality_period)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def sku_optimizer(skus: list[dict]) -> dict:
    """Classify SKUs using ABC/XYZ analysis and recommend inventory strategies.

    Args:
        skus: List of SKUs as [{"sku": "ABC123", "revenue": 50000, "quantity": 200, "cost": 50, "demand_cv": 0.3}]
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _sku_optimizer(skus)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def warehouse_layout(zones: list[dict], total_sqft: float = 10000,
                     picking_method: str = "zone") -> dict:
    """Plan warehouse zone layout optimized for picking efficiency.

    Args:
        zones: Warehouse zones as [{"name": "Electronics", "sku_count": 200, "daily_picks": 500, "storage_type": "shelf"}]
        total_sqft: Total warehouse square footage
        picking_method: Picking strategy (wave, zone, batch, discrete)
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _warehouse_layout(zones, total_sqft, picking_method)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def shrinkage_detector(inventory_records: list[dict]) -> dict:
    """Detect inventory shrinkage by comparing expected vs actual quantities.
    Flags anomalies and provides category-level analysis.

    Args:
        inventory_records: Records as [{"sku": "X", "expected_qty": 100, "actual_qty": 95, "value_per_unit": 25.0, "category": "electronics"}]
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _shrinkage_detector(inventory_records)
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()
