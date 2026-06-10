import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import charts
from utils.helpers import (
    breakdown,
    compute_kpis,
    compute_rfm,
    discount_bands,
    monthly_trend,
    rfm_clusters,
    yoy_growth,
)

PALETTE = ["#0068C9", "#83C9FF", "#FF8C00", "#29B09D", "#7D3C98", "#E45756"]


def _apply(fig, height=400, bottom_extra=0):
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else ""
    fig.update_layout(
        height=height,
        margin=dict(l=56, r=40, t=72, b=72 + bottom_extra),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(family="Source Sans 3, sans-serif", color="#31333F", size=12),
        title=dict(text=title, font=dict(size=14), x=0, xanchor="left", y=0.98, yanchor="top", pad=dict(b=12)),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, xanchor="left", font=dict(size=11), title_text=""),
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True),
    )
    return fig


def chart_kpi_summary(df):
    kpis = compute_kpis(df)
    fig = go.Figure(go.Bar(
        x=["Sales ($K)", "Profit ($K)", "Margin (%)"],
        y=[kpis["sales"] / 1000, kpis["profit"] / 1000, kpis["margin"]],
        marker_color=["#0068C9", "#29B09D", "#FF8C00"],
        text=[f"${kpis['sales']/1000:.0f}K", f"${kpis['profit']/1000:.0f}K", f"{kpis['margin']:.1f}%"],
        textposition="outside",
    ))
    fig.update_layout(title="Overall KPI Snapshot", yaxis_title="Value")
    return _apply(fig, 380)


def chart_category_sales_vs_profit_share(df):
    cat = breakdown(df, "Category")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Sales Share %", x=cat["Category"], y=cat["Sales Share %"], marker_color="#0068C9"))
    fig.add_trace(go.Bar(name="Profit Share %", x=cat["Category"], y=cat["Profit Share %"], marker_color="#29B09D"))
    fig.update_layout(title="Sales Share vs Profit Share by Category", barmode="group", yaxis_title="Share (%)")
    return _apply(fig, 400)


def chart_category_margin(df):
    cat = breakdown(df, "Category").sort_values("Margin %", ascending=True)
    colors = ["#29B09D" if m >= 13 else "#E45756" if m < 10 else "#FF8C00" for m in cat["Margin %"]]
    fig = go.Figure(go.Bar(
        x=cat["Margin %"], y=cat["Category"], orientation="h",
        marker_color=colors, text=[f"{v:.1f}%" for v in cat["Margin %"]], textposition="outside",
    ))
    fig.update_layout(title="Profit Margin by Category", xaxis_title="Margin (%)")
    return _apply(fig, 360)


def chart_loss_subcategories(df):
    sub = breakdown(df, "Sub-Category")
    loss = sub[sub["Profit"] < 0].sort_values("Profit")
    if loss.empty:
        return None
    fig = go.Figure(go.Bar(
        x=loss["Profit"], y=loss["Sub-Category"], orientation="h",
        marker_color="#E45756",
        text=[f"${v:,.0f}" for v in loss["Profit"]], textposition="outside",
    ))
    fig.update_layout(title="Loss-Making Sub-Categories", xaxis_title="Profit ($)")
    return _apply(fig, 380)


def chart_top_vs_loss_states(df):
    state = df.groupby("State", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    top = state.nlargest(5, "Sales")
    loss = state.nsmallest(5, "Profit")
    combined = pd.concat([top, loss]).drop_duplicates("State").sort_values("Profit")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Sales", x=combined["State"], y=combined["Sales"], marker_color="#0068C9"))
    fig.add_trace(go.Bar(
        name="Profit", x=combined["State"], y=combined["Profit"],
        marker_color=["#29B09D" if p >= 0 else "#E45756" for p in combined["Profit"]],
    ))
    fig.update_layout(title="High-Sales vs Low-Profit States", barmode="group", xaxis_tickangle=-30)
    return _apply(fig, 420)


def chart_top_states(df):
    state = df.groupby("State", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    top = state.nlargest(8, "Sales")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top["State"], y=top["Sales"], name="Sales", marker_color="#0068C9"))
    fig.add_trace(go.Bar(
        x=top["State"], y=top["Profit"], name="Profit",
        marker_color=["#29B09D" if p >= 0 else "#E45756" for p in top["Profit"]],
    ))
    fig.update_layout(title="Top States by Sales & Profit", barmode="group", xaxis_tickangle=-30)
    return _apply(fig, 400)


def chart_customer_leaders(df):
    cust = df.groupby(["Customer ID", "Customer Name"], as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    top = cust.nlargest(8, "Sales")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top["Customer Name"], y=top["Sales"], name="Sales", marker_color="#0068C9"))
    fig.add_trace(go.Bar(
        x=top["Customer Name"], y=top["Profit"], name="Profit",
        marker_color=["#29B09D" if p >= 0 else "#E45756" for p in top["Profit"]],
    ))
    fig.update_layout(title="Top Customers: Revenue vs Profit", barmode="group", xaxis_tickangle=-25)
    return _apply(fig, 440, bottom_extra=20)


def chart_customer_concentration(df):
    cust = df.groupby("Customer ID", as_index=False).agg(Sales=("Sales", "sum"))
    sorted_c = cust.sort_values("Sales", ascending=False).reset_index(drop=True)
    sorted_c["Customer Rank %"] = (sorted_c.index + 1) / len(sorted_c) * 100
    sorted_c["Cumulative Sales %"] = sorted_c["Sales"].cumsum() / sorted_c["Sales"].sum() * 100
    fig = px.line(
        sorted_c, x="Customer Rank %", y="Cumulative Sales %",
        title="Customer Concentration Curve",
        labels={"Customer Rank %": "Customers (cumulative %)", "Cumulative Sales %": "Revenue (cumulative %)"},
    )
    fig.add_hline(y=80, line_dash="dash", line_color="#FF8C00", annotation_text="80% revenue")
    return _apply(fig, 400)


def chart_rfm_cluster_counts(df, n_clusters):
    rfm = rfm_clusters(compute_rfm(df), n_clusters)
    counts = rfm.groupby("Cluster Label", observed=True).size().reset_index(name="Customers")
    fig = px.bar(
        counts, x="Cluster Label", y="Customers", color="Cluster Label",
        color_discrete_sequence=PALETTE, title="RFM Cluster Distribution",
    )
    fig.update_layout(showlegend=False)
    return _apply(fig, 380)


CHART_RENDERERS = {
    "kpi_summary": lambda df, n: chart_kpi_summary(df),
    "monthly_trend": lambda df, n: charts.line_sales_profit(monthly_trend(df)),
    "yoy_growth": lambda df, n: charts.bar_yoy(yoy_growth(df)),
    "segment_breakdown": lambda df, n: charts.donut_breakdown(breakdown(df, "Segment"), "Segment", "Sales by Segment"),
    "region_margin": lambda df, n: charts.bar_region_margin(breakdown(df, "Region")),
    "top_loss_states": lambda df, n: chart_top_vs_loss_states(df),
    "top_states": lambda df, n: chart_top_states(df),
    "category_share_trap": lambda df, n: chart_category_sales_vs_profit_share(df),
    "loss_subcategories": lambda df, n: chart_loss_subcategories(df),
    "category_margin": lambda df, n: chart_category_margin(df),
    "discount_bands": lambda df, n: charts.bar_discount_bands(discount_bands(df)),
    "customer_leaders": lambda df, n: chart_customer_leaders(df),
    "customer_concentration": lambda df, n: chart_customer_concentration(df),
    "rfm_clusters": lambda df, n: charts.scatter_rfm_clusters(rfm_clusters(compute_rfm(df), n)),
    "rfm_cluster_counts": lambda df, n: chart_rfm_cluster_counts(df, n),
}


def render_insight_chart(chart_key: str, df, n_clusters: int = 4):
    renderer = CHART_RENDERERS.get(chart_key)
    if not renderer:
        return None
    try:
        return renderer(df, n_clusters)
    except Exception:
        return None
