import pandas as pd
import streamlit as st

from utils import charts
from utils.helpers import (
    apply_theme,
    breakdown,
    compute_kpis,
    compute_rfm,
    dataset_description_table,
    discount_bands,
    fmt_currency,
    insight,
    load_data,
    page_title,
    render_metrics,
    render_sidebar,
    render_insights_tab,
    rfm_clusters,
    section_header,
    styled_dataframe,
    summary_statistics_table,
    monthly_trend,
    yoy_growth,
)

st.set_page_config(page_title="Superstore Analytics Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
apply_theme()


def tab_overview(df):
    kpis = compute_kpis(df)
    page_title("Superstore Sales Overview", "Big Data Final Report · Sales analysis 2014–2017")
    render_metrics([
        ("Total Sales", fmt_currency(kpis["sales"]), None),
        ("Total Profit", fmt_currency(kpis["profit"]), None),
        ("Profit Margin", f"{kpis['margin']:.1f}%", None),
        ("Orders", f"{kpis['orders']:,}", None),
        ("Customers", f"{kpis['customers']:,}", f"Loss rate {kpis['loss_rate']:.1f}%"),
    ])
    section_header("Dataset Description")
    styled_dataframe(dataset_description_table())
    section_header("Summary Statistics")
    styled_dataframe(summary_statistics_table(kpis, df))
    insight("Superstore generated ~$2.3M in sales and ~$286K in profit (margin ~12.5%). Profit growth is beginning to lag behind revenue growth.")
    section_header("Sales & Profit Trends")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.line_sales_profit(monthly_trend(df)), use_container_width=True, key="overview_monthly_trend")
    with c2:
        st.plotly_chart(charts.bar_yoy(yoy_growth(df)), use_container_width=True, key="overview_yoy_growth")
    section_header("Revenue Breakdown")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(charts.donut_breakdown(breakdown(df, "Region"), "Region", "Sales by Region"), use_container_width=True, key="overview_donut_region")
    with c2:
        st.plotly_chart(charts.donut_breakdown(breakdown(df, "Category"), "Category", "Sales by Category"), use_container_width=True, key="overview_donut_category")
    with c3:
        st.plotly_chart(charts.donut_breakdown(breakdown(df, "Segment"), "Segment", "Sales by Segment"), use_container_width=True, key="overview_donut_segment")


def tab_geography(df):
    page_title("Geographic Performance", "Regional and state-level analysis")
    region_data = breakdown(df, "Region")
    state_data = df.groupby(["State", "State Abbr"], as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    section_header("Regional Summary")
    styled_dataframe(region_data[["Region", "Sales", "Profit", "Margin %", "Sales Share %", "Profit Share %"]].style.format(
        {"Sales": "${:,.0f}", "Profit": "${:,.0f}", "Margin %": "{:.1f}%", "Sales Share %": "{:.1f}%", "Profit Share %": "{:.1f}%"}))
    section_header("Profit by State")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(charts.choropleth_states(state_data), use_container_width=True, key="geo_choropleth")
    with c2:
        st.plotly_chart(charts.bar_region_margin(region_data), use_container_width=True, key="geo_region_margin")
    section_header("Top & Bottom States")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(charts.bar_states(state_data, "Top 10 States by Sales & Profit"), use_container_width=True, key="geo_top_states")
    with c4:
        st.plotly_chart(charts.bar_states(state_data.nsmallest(10, "Profit"), "Bottom 10 States by Profit", ascending=True), use_container_width=True, key="geo_bottom_states")


def tab_product(df):
    page_title("Product & Sales Performance", "Category profitability and discount diagnostics")
    cat_data, sub_data = breakdown(df, "Category"), breakdown(df, "Sub-Category")
    section_header("Category Summary")
    styled_dataframe(cat_data.style.format({"Sales": "${:,.0f}", "Profit": "${:,.0f}", "Margin %": "{:.1f}%", "Sales Share %": "{:.1f}%", "Profit Share %": "{:.1f}%"}))
    section_header("Profit by Sub-Category")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.bar_subcategory(sub_data), use_container_width=True, key="product_subcategory")
    with c2:
        st.plotly_chart(charts.treemap_category(df.groupby(["Category", "Sub-Category"], as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))), use_container_width=True, key="product_treemap")
    section_header("Discount Diagnostics")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(charts.scatter_discount(df), use_container_width=True, key="product_discount_scatter")
    with c4:
        st.plotly_chart(charts.bar_discount_bands(discount_bands(df)), use_container_width=True, key="product_discount_bands")


def tab_customer(df):
    page_title("Customer Analysis", "Customer value and RFM segmentation")
    cust = df.groupby(["Customer ID", "Customer Name"], as_index=False).agg(Orders=("Order ID", "nunique"), Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    section_header("Top Customers")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(charts.bar_top_customers(cust.nlargest(10, "Orders").rename(columns={"Orders": "Frequency"}), "Frequency", "Top 10 by Orders"), use_container_width=True, key="customer_top_orders")
    with c2:
        st.plotly_chart(charts.bar_top_customers(cust.nlargest(10, "Sales"), "Sales", "Top 10 by Revenue"), use_container_width=True, key="customer_top_sales")
    with c3:
        st.plotly_chart(charts.bar_top_customers(cust.nlargest(10, "Profit"), "Profit", "Top 10 by Profit"), use_container_width=True, key="customer_top_profit")
    section_header("RFM Analysis")
    st.plotly_chart(charts.scatter_rfm(compute_rfm(df)), use_container_width=True, key="customer_rfm_scatter")


def tab_advanced(df, n_clusters):
    page_title("Advanced Analysis", "RFM clustering vs marketing segments")
    rfm = rfm_clusters(compute_rfm(df), n_clusters)
    section_header("Cluster Visualization")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.scatter_rfm_clusters(rfm), use_container_width=True, key="advanced_rfm_clusters")
    with c2:
        st.plotly_chart(charts.bar_cluster_vs_segment(rfm), use_container_width=True, key="advanced_cluster_segment")


df_raw = load_data()
df, n_clusters = render_sidebar(df_raw)

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

tabs = st.tabs(["Overview", "Geography", "Product & Sales", "Customer", "Advanced (RFM)", "Insights"])
with tabs[0]:
    tab_overview(df)
with tabs[1]:
    tab_geography(df)
with tabs[2]:
    tab_product(df)
with tabs[3]:
    tab_customer(df)
with tabs[4]:
    tab_advanced(df, n_clusters)
with tabs[5]:
    render_insights_tab(df, n_clusters)
