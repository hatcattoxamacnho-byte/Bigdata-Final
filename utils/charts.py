import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#0068C9", "#83C9FF", "#FF8C00", "#29B09D", "#7D3C98", "#E45756"]


def _theme(fig, height=420, bottom_extra=0):
    """Apply consistent layout — title on top, legend below chart area to avoid overlap."""
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else ""
    bottom = 72 + bottom_extra
    fig.update_layout(
        height=height,
        margin=dict(l=56, r=40, t=72, b=bottom),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3, sans-serif", color="#31333F", size=12),
        title=dict(
            text=title,
            font=dict(size=14),
            x=0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            pad=dict(t=8, b=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22 - (bottom_extra / height),
            x=0,
            xanchor="left",
            font=dict(size=11),
            title_text="",
            tracegroupgap=12,
            itemsizing="constant",
        ),
        xaxis=dict(showgrid=True, gridcolor="#E6E9EF", zeroline=False, automargin=True),
        yaxis=dict(showgrid=True, gridcolor="#E6E9EF", zeroline=False, automargin=True),
    )
    return fig


def _theme_pie(fig, height=400):
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else ""
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=72, b=90),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3, sans-serif", color="#31333F", size=12),
        title=dict(text=title, font=dict(size=14), x=0.5, xanchor="center", y=0.97, yanchor="top", pad=dict(b=12)),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            x=0.5,
            xanchor="center",
            font=dict(size=11),
            title_text="",
        ),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    fig.update_traces(textinfo="percent", textposition="inside", insidetextorientation="horizontal")
    return fig


def line_sales_profit(monthly):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["YearMonth"], y=monthly["Sales"], name="Sales", mode="lines+markers", line=dict(color="#0068C9")))
    fig.add_trace(go.Scatter(x=monthly["YearMonth"], y=monthly["Profit"], name="Profit", mode="lines+markers", line=dict(color="#FF8C00"), yaxis="y2"))
    fig.update_layout(title="Sales & Profit by Month (2014–2017)", yaxis=dict(title="Sales ($)"), yaxis2=dict(title="Profit ($)", overlaying="y", side="right"))
    return _theme(fig, 460)


def bar_yoy(yoy):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yoy["Year"].astype(str), y=yoy["Sales Growth %"], name="Sales Growth %", marker_color="#0068C9"))
    fig.add_trace(go.Bar(x=yoy["Year"].astype(str), y=yoy["Profit Growth %"], name="Profit Growth %", marker_color="#FF8C00"))
    fig.update_layout(title="YoY Growth: Sales vs Profit", barmode="group")
    return _theme(fig, 400)


def donut_breakdown(data, name_col, title):
    fig = px.pie(data, names=name_col, values="Sales", hole=0.55, color_discrete_sequence=PALETTE, title=title)
    return _theme_pie(fig, 400)


def bar_region_margin(data):
    data = data.sort_values("Margin %", ascending=True)
    colors = ["#29B09D" if m >= 13 else "#E45756" if m < 10 else "#FF8C00" for m in data["Margin %"]]
    fig = go.Figure(go.Bar(
        x=data["Margin %"], y=data["Region"], orientation="h",
        marker_color=colors, text=[f"{v:.1f}%" for v in data["Margin %"]], textposition="outside",
    ))
    fig.update_layout(title="Profit Margin by Region", xaxis_title="Margin (%)")
    return _theme(fig, 380)


def choropleth_states(state_data):
    fig = px.choropleth(
        state_data, locations="State Abbr", locationmode="USA-states", color="Profit", scope="usa",
        color_continuous_scale=["#E45756", "#F3F4F6", "#29B09D"], hover_name="State",
        hover_data={"Sales": ":,.0f", "Profit": ":,.0f", "State Abbr": False},
        title="Profit by State",
    )
    fig.update_layout(
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        coloraxis_colorbar=dict(title="Profit", len=0.7, thickness=12),
        margin=dict(l=0, r=0, t=72, b=20),
        title=dict(font=dict(size=14), x=0, xanchor="left", y=0.98, yanchor="top"),
        height=500,
    )
    return fig


def bar_states(data, title, ascending=False, n=10):
    data = data.sort_values("Sales", ascending=ascending).head(n)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data["State"], y=data["Sales"], name="Sales", marker_color="#0068C9"))
    fig.add_trace(go.Bar(
        x=data["State"], y=data["Profit"], name="Profit",
        marker_color=["#29B09D" if p >= 0 else "#E45756" for p in data["Profit"]],
    ))
    fig.update_layout(title=title, barmode="group", xaxis_tickangle=-35)
    return _theme(fig, 440)


def bar_subcategory(data):
    data = data.sort_values("Profit", ascending=True)
    fig = go.Figure(go.Bar(
        x=data["Profit"], y=data["Sub-Category"], orientation="h",
        marker_color=["#29B09D" if p >= 0 else "#E45756" for p in data["Profit"]],
        text=[f"${v:,.0f}" for v in data["Profit"]], textposition="outside",
    ))
    fig.update_layout(title="Profit by Sub-Category", xaxis_title="Profit ($)")
    return _theme(fig, 560)


def treemap_category(data):
    fig = px.treemap(
        data, path=["Category", "Sub-Category"], values="Sales", color="Profit",
        color_continuous_scale=["#E45756", "#F3F4F6", "#29B09D"],
        title="Treemap: Category → Sub-Category",
    )
    fig.update_layout(
        height=480,
        margin=dict(l=8, r=8, t=72, b=8),
        title=dict(font=dict(size=14), x=0, xanchor="left", y=0.98, yanchor="top"),
        coloraxis_colorbar=dict(title="Profit", len=0.75),
    )
    return fig


def scatter_discount(df):
    sample = df.sample(min(len(df), 1500), random_state=42)
    fig = px.scatter(
        sample, x="Discount", y="Profit", color="Category",
        opacity=0.55, color_discrete_sequence=PALETTE, title="Discount vs Profit",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#9CA3AF")
    return _theme(fig, 460, bottom_extra=20)


def bar_discount_bands(bands):
    fig = go.Figure(go.Bar(
        x=bands["Discount Band"].astype(str), y=bands["Avg_Profit"],
        marker_color=["#29B09D" if p >= 0 else "#E45756" for p in bands["Avg_Profit"]],
        text=[f"${v:,.0f}" for v in bands["Avg_Profit"]], textposition="outside",
    ))
    fig.update_layout(title="Avg Profit by Discount Band", yaxis_title="Avg Profit / line ($)")
    return _theme(fig, 400)


def bar_top_customers(data, metric, title):
    data = data.sort_values(metric, ascending=True).tail(10)
    fig = go.Figure(go.Bar(
        x=data[metric], y=data["Customer Name"], orientation="h",
        marker_color=["#0068C9" if (metric != "Profit" or v >= 0) else "#E45756" for v in data[metric]],
        text=[f"${v:,.0f}" if metric != "Frequency" else f"{int(v)}" for v in data[metric]],
        textposition="outside",
    ))
    fig.update_layout(title=title)
    return _theme(fig, 420)


def scatter_rfm(rfm):
    fig = px.scatter(
        rfm, x="Frequency", y="Monetary", size="Recency", color="Profit",
        hover_name="Customer Name", opacity=0.7,
        color_continuous_scale=["#E45756", "#F3F4F6", "#29B09D"],
        title="RFM Analysis (size = Recency, color = Profit)",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Profit", len=0.75))
    return _theme(fig, 480, bottom_extra=10)


def scatter_rfm_clusters(rfm):
    fig = px.scatter(
        rfm, x="Frequency", y="Monetary", color="Cluster Label",
        size="Recency", hover_name="Customer Name", opacity=0.75,
        color_discrete_sequence=PALETTE, title="RFM Clustering (K-Means)",
    )
    return _theme(fig, 480, bottom_extra=24)


def bar_cluster_vs_segment(rfm):
    cross = rfm.groupby(["Cluster Label", "Segment"], observed=True).size().reset_index(name="Count")
    fig = px.bar(
        cross, x="Cluster Label", y="Count", color="Segment",
        barmode="group", color_discrete_sequence=PALETTE, title="RFM Cluster vs Segment",
    )
    return _theme(fig, 440, bottom_extra=24)
