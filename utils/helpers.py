from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Superstore.csv"

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

PROFESSIONAL_CSS = """
<style>
.stApp { color: #31333F; }
.main .block-container { padding-top: 2.5rem !important; padding-bottom: 3rem; max-width: 1180px; }
h1 { color: #31333F !important; font-size: 2.25rem !important; font-weight: 700 !important; }
h3 { color: #31333F !important; font-size: 1.125rem !important; font-weight: 600 !important;
     margin-top: 1.75rem !important; border-bottom: 1px solid #E6E9EF; padding-bottom: 0.35rem; }
div[data-testid="stMetric"] { background: #FFF; border: 1px solid #E6E9EF; border-radius: 8px; padding: 1rem; }
div[data-testid="stSidebar"] { background: #FAFAFA; border-right: 1px solid #E6E9EF; }

/* Tabs were hidden under the fixed Streamlit header — push them down */
div[data-testid="stTabs"] { margin-top: 1.5rem !important; }
.stTabs [data-baseweb="tab-list"] { overflow: visible !important; height: auto !important; min-height: 48px !important; border-bottom: 1px solid #E6E9EF; }
.stTabs button[data-baseweb="tab"] { height: auto !important; min-height: 44px !important; padding: 0.6rem 1.2rem !important; line-height: 1.4 !important; overflow: visible !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #FF4B4B !important; }

.insight-text { color: #4B5563; font-size: 0.925rem; line-height: 1.6; margin: 0.5rem 0 1.25rem 0; }

/* Prevent Plotly chart titles/legends from being clipped */
[data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] > div, .stPlotlyChart {
    overflow: visible !important;
}
[data-testid="column"] { overflow: visible !important; }

.insight-card {
    background: #FFF; border: 1px solid #E6E9EF; border-radius: 8px;
    padding: 1rem 1.2rem; margin-bottom: 0.75rem;
}
.insight-card h4 { margin: 0 0 0.4rem 0; font-size: 0.95rem; color: #31333F; }
.insight-card p { margin: 0.25rem 0; font-size: 0.875rem; color: #4B5563; line-height: 1.5; }
.insight-card .label { font-weight: 600; color: #31333F; }
.badge-critical { display:inline-block; background:#FEE2E2; color:#B91C1C; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-warning { display:inline-block; background:#FFEDD5; color:#C2410C; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-opportunity { display:inline-block; background:#D1FAE5; color:#047857; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge-info { display:inline-block; background:#DBEAFE; color:#1D4ED8; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }

/* Insight expanders — grouped accordion layout */
.insights-group-title {
    font-size: 1.05rem; font-weight: 600; color: #31333F;
    margin: 1.5rem 0 0.65rem 0; padding-bottom: 0.35rem; border-bottom: 1px solid #E6E9EF;
}
div[data-testid="stExpander"] {
    background: #FFF; border: 1px solid #E6E9EF !important; border-radius: 8px !important;
    margin-bottom: 0.5rem !important; overflow: hidden;
}
div[data-testid="stExpander"] summary {
    font-size: 0.925rem !important; font-weight: 500 !important; color: #31333F !important;
    padding: 0.65rem 0.85rem !important;
}
div[data-testid="stExpander"] summary:hover { background: #F8F9FB; }
div[data-testid="stExpander"] .streamlit-expanderContent {
    padding: 0 1rem 1rem 1rem !important; border-top: 1px solid #EEF0F4;
}
.insight-detail-label { font-weight: 600; color: #31333F; font-size: 0.875rem; }
.insight-detail-text { color: #4B5563; font-size: 0.875rem; line-height: 1.55; margin: 0.15rem 0 0.75rem 0; }
.tag-info { color: #1D4ED8; font-weight: 600; }
.tag-warning { color: #C2410C; font-weight: 600; }
.tag-critical { color: #B91C1C; font-weight: 600; }
.tag-opportunity { color: #047857; font-weight: 600; }
</style>
"""


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Order Date"] = pd.to_datetime(out["Order Date"], format="%m/%d/%Y")
    out["Ship Date"] = pd.to_datetime(out["Ship Date"], format="%m/%d/%Y")
    out["Year"] = out["Order Date"].dt.year
    out["YearMonth"] = out["Order Date"].dt.strftime("%Y-%m")
    out["State Abbr"] = out["State"].map(STATE_ABBR)
    out["Is Loss"] = out["Profit"] < 0
    return out


@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    return enrich_dataframe(pd.read_csv(DATA_PATH, encoding="latin-1"))


def filter_data(df, years=None, regions=None, segments=None, categories=None):
    filtered = df.copy()
    if years:
        filtered = filtered[filtered["Year"].isin(years)]
    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]
    if segments:
        filtered = filtered[filtered["Segment"].isin(segments)]
    if categories:
        filtered = filtered[filtered["Category"].isin(categories)]
    return filtered


def compute_kpis(df):
    sales, profit = df["Sales"].sum(), df["Profit"].sum()
    return {
        "sales": sales,
        "profit": profit,
        "margin": (profit / sales * 100) if sales else 0,
        "orders": df["Order ID"].nunique(),
        "customers": df["Customer ID"].nunique(),
        "loss_rate": df["Is Loss"].mean() * 100,
    }


def monthly_trend(df):
    return df.groupby("YearMonth", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).sort_values("YearMonth")


def yoy_growth(df):
    yearly = df.groupby("Year", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    yearly["Sales Growth %"] = yearly["Sales"].pct_change() * 100
    yearly["Profit Growth %"] = yearly["Profit"].pct_change() * 100
    return yearly


def breakdown(df, column):
    result = df.groupby(column, as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    ts, tp = result["Sales"].sum(), result["Profit"].sum()
    result["Sales Share %"] = result["Sales"] / ts * 100 if ts else 0
    result["Profit Share %"] = result["Profit"] / tp * 100 if tp else 0
    result["Margin %"] = np.where(result["Sales"] > 0, result["Profit"] / result["Sales"] * 100, 0)
    return result.sort_values("Sales", ascending=False)


def discount_bands(df):
    banded = df.copy()
    banded["Discount Band"] = pd.cut(
        banded["Discount"], bins=[-0.001, 0.001, 0.2, 0.4, 0.6, 0.81],
        labels=["0%", "1–20%", "21–40%", "41–60%", "61–80%"], include_lowest=True,
    )
    return banded.groupby("Discount Band", observed=True).agg(Avg_Profit=("Profit", "mean")).reset_index()


def compute_rfm(df, reference_date=None):
    ref = reference_date or df["Order Date"].max() + pd.Timedelta(days=1)
    return df.groupby(["Customer ID", "Customer Name", "Region", "Segment"], as_index=False).agg(
        Recency=("Order Date", lambda x: (ref - x.max()).days),
        Frequency=("Order ID", "nunique"),
        Monetary=("Sales", "sum"),
        Profit=("Profit", "sum"),
    )


def rfm_clusters(rfm, n_clusters=4):
    scaled = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)
    result = rfm.copy()
    result["Cluster"] = labels
    names = {}
    for c in sorted(result["Cluster"].unique()):
        s = result[result["Cluster"] == c]
        r, f, m = s["Recency"].mean(), s["Frequency"].mean(), s["Monetary"].mean()
        if f >= result["Frequency"].median() and m >= result["Monetary"].median() and r <= result["Recency"].median():
            names[c] = "Champions"
        elif r <= result["Recency"].median() and f >= result["Frequency"].median():
            names[c] = "Loyal"
        elif r > result["Recency"].median() and m >= result["Monetary"].median():
            names[c] = "At Risk"
        else:
            names[c] = "Developing"
    result["Cluster Label"] = result["Cluster"].map(names)
    return result


def fmt_currency(v):
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:.1f}K"
    return f"${v:,.0f}"


def apply_theme():
    st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)


def page_title(title, subtitle=""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def section_header(title):
    st.markdown(f"### {title}")


def insight(text):
    st.markdown(f'<p class="insight-text">{text}</p>', unsafe_allow_html=True)


def render_metrics(metrics):
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.metric(label, value, delta) if delta is not None else st.metric(label, value)


def styled_dataframe(df):
    st.dataframe(df, use_container_width=True, hide_index=True)


def dataset_description_table():
    return pd.DataFrame({
        "Column": ["Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Customer Name", "Segment",
                   "Country", "City", "State", "Postal Code", "Region", "Product ID", "Category", "Sub-Category",
                   "Product Name", "Sales", "Quantity", "Discount", "Profit"],
        "Description": [
            "Unique order identifier", "Date the order was placed", "Date the order was shipped", "Shipping service level",
            "Unique customer identifier", "Customer full name", "Customer segment",
            "Country of sale", "City of sale", "US state of sale", "Postal / ZIP code",
            "Sales region", "Unique product identifier", "Product category", "Product sub-category",
            "Product name", "Revenue ($)", "Units sold", "Discount rate (0–1)", "Profit ($)",
        ],
    })


def summary_statistics_table(kpis, df):
    return pd.DataFrame({
        "Metric": ["Total Sales", "Total Profit", "Profit Margin", "Total Orders", "Total Customers",
                   "Line Items", "Loss-Making Lines (%)", "Date Range"],
        "Value": [
            fmt_currency(kpis["sales"]), fmt_currency(kpis["profit"]), f"{kpis['margin']:.2f}%",
            f"{kpis['orders']:,}", f"{kpis['customers']:,}", f"{len(df):,}", f"{kpis['loss_rate']:.1f}%",
            f"{df['Order Date'].min():%b %Y} – {df['Order Date'].max():%b %Y}",
        ],
    })


def render_sidebar(df):
    st.sidebar.markdown("## Filters")
    years = st.sidebar.multiselect("Year", sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))
    regions = st.sidebar.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    segments = st.sidebar.multiselect("Segment", sorted(df["Segment"].unique()), default=sorted(df["Segment"].unique()))
    categories = st.sidebar.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    st.sidebar.markdown("---")
    n_clusters = st.sidebar.slider("RFM Clusters (K-Means)", 2, 6, 4)
    return filter_data(df, years, regions, segments, categories), n_clusters


INSIGHT_GROUPS = ["Overview", "Geography", "Product & Sales", "Customer", "Advanced (RFM)"]

PRIORITY_BADGE = {
    "Critical": "badge-critical",
    "Warning": "badge-warning",
    "Opportunity": "badge-opportunity",
    "Info": "badge-info",
}


def _insight(group, title, finding, recommendation, chart_key, priority="Info"):
    return {
        "Group": group, "Insight": title, "Finding": finding,
        "Recommendation": recommendation, "Priority": priority, "ChartKey": chart_key,
    }


def generate_insights(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    rows = []
    kpis = compute_kpis(df)
    region = breakdown(df, "Region")
    category = breakdown(df, "Category")
    subcat = breakdown(df, "Sub-Category")
    yearly = yoy_growth(df)

    rows.append(_insight(
        "Overview", "Overall Business Performance",
        f"Total sales {fmt_currency(kpis['sales'])}, profit {fmt_currency(kpis['profit'])}, margin {kpis['margin']:.1f}%. "
        f"{kpis['orders']:,} orders from {kpis['customers']:,} customers.",
        "Monitor profit quality alongside revenue growth; do not rely on sales volume alone.",
        "monthly_trend", "Info",
    ))

    if len(yearly) >= 2:
        last = yearly.iloc[-1]
        prev = yearly.iloc[-2]
        if pd.notna(last["Sales Growth %"]) and pd.notna(last["Profit Growth %"]):
            if last["Profit Growth %"] < last["Sales Growth %"]:
                rows.append(_insight(
                    "Overview", "Profit Growth Lagging Sales",
                    f"In {int(last['Year'])}, sales grew {last['Sales Growth %']:.1f}% but profit grew only {last['Profit Growth %']:.1f}%.",
                    "Investigate discounting, product mix, and cost structure driving margin erosion.",
                    "yoy_growth", "Warning",
                ))

    seg = breakdown(df, "Segment")
    top_seg = seg.iloc[0]
    rows.append(_insight(
        "Overview", "Revenue by Segment",
        f"{top_seg['Segment']} is the largest segment ({top_seg['Sales Share %']:.1f}% of sales, margin {top_seg['Margin %']:.1f}%).",
        "Align marketing spend with segment profitability, not just segment volume.",
        "segment_breakdown", "Info",
    ))

    if not region.empty:
        best = region.loc[region["Margin %"].idxmax()]
        worst = region.loc[region["Margin %"].idxmin()]
        rows.append(_insight(
            "Geography", "Regional Margin Gap",
            f"{best['Region']} leads with {best['Margin %']:.1f}% margin; {worst['Region']} trails at {worst['Margin %']:.1f}%.",
            f"Review pricing, discount policy, and product mix in {worst['Region']}.",
            "region_margin", "Warning" if worst["Margin %"] < 10 else "Info",
        ))

    state = df.groupby("State", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    loss_states = state[state["Profit"] < 0].sort_values("Profit")
    if not loss_states.empty:
        s = loss_states.iloc[0]
        rows.append(_insight(
            "Geography", "High-Sales, Low-Profit States",
            f"{s['State']} generated {fmt_currency(s['Sales'])} in sales but {fmt_currency(s['Profit'])} in profit.",
            "Audit unprofitable states for discount abuse and loss-making SKUs.",
            "top_loss_states", "Critical",
        ))

    top_states = state.nlargest(2, "Sales")
    if len(top_states) >= 1:
        t = top_states.iloc[0]
        rows.append(_insight(
            "Geography", "Top Market",
            f"{t['State']} is the #1 state by sales ({fmt_currency(t['Sales'])}) with profit {fmt_currency(t['Profit'])}.",
            "Protect and expand share in top-performing markets.",
            "top_states", "Opportunity",
        ))

    furn = category[category["Category"] == "Furniture"]
    if not furn.empty:
        f = furn.iloc[0]
        rows.append(_insight(
            "Product & Sales", "Furniture Profitability Trap",
            f"Furniture contributes {f['Sales Share %']:.1f}% of sales but only {f['Profit Share %']:.1f}% of profit (margin {f['Margin %']:.1f}%).",
            "Restructure furniture portfolio; reduce exposure to low-margin sub-categories.",
            "category_share_trap", "Critical",
        ))

    loss_subs = subcat[subcat["Profit"] < 0].sort_values("Profit")
    if not loss_subs.empty:
        names = ", ".join(f"{r['Sub-Category']} ({fmt_currency(r['Profit'])})" for _, r in loss_subs.head(3).iterrows())
        rows.append(_insight(
            "Product & Sales", "Loss-Making Sub-Categories",
            f"Net-loss sub-categories include: {names}.",
            "Reprice, reduce inventory, or discontinue persistently unprofitable SKUs.",
            "loss_subcategories", "Critical",
        ))

    tech = category[category["Category"] == "Technology"]
    if not tech.empty:
        t = tech.iloc[0]
        rows.append(_insight(
            "Product & Sales", "Technology Drives Profit",
            f"Technology delivers {t['Margin %']:.1f}% margin and {t['Profit Share %']:.1f}% of total profit.",
            "Prioritize investment and inventory in high-margin Technology and Office Supplies.",
            "category_margin", "Opportunity",
        ))

    rows.append(_insight(
        "Product & Sales", "Discount Impact",
        f"{kpis['loss_rate']:.1f}% of line items are sold at a loss. Profit turns negative when discounts exceed ~20%.",
        "Implement discount governance; cap standard discounts at approximately 20%.",
        "discount_bands", "Warning",
    ))

    cust = df.groupby(["Customer ID", "Customer Name"], as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    top_rev = cust.nlargest(1, "Sales").iloc[0]
    top_profit = cust.nlargest(1, "Profit").iloc[0]
    rows.append(_insight(
        "Customer", "Revenue vs Profit Leaders",
        f"Top revenue customer: {top_rev['Customer Name']} ({fmt_currency(top_rev['Sales'])}). "
        f"Top profit customer: {top_profit['Customer Name']} ({fmt_currency(top_profit['Profit'])}).",
        "Focus retention on high-profit customers, not just high-revenue accounts.",
        "customer_leaders", "Warning" if top_rev["Profit"] < 0 else "Info",
    ))

    if top_rev["Customer Name"] != top_profit["Customer Name"] and top_rev["Profit"] < 0:
        rows.append(_insight(
            "Customer", "Revenue Does Not Equal Value",
            f"{top_rev['Customer Name']} is the highest-revenue customer but records a loss of {fmt_currency(top_rev['Profit'])}.",
            "Renegotiate pricing or reduce discounting for high-volume, unprofitable accounts.",
            "customer_leaders", "Critical",
        ))

    sorted_cust = cust.sort_values("Sales", ascending=False).reset_index(drop=True)
    sorted_cust["CumSalesPct"] = sorted_cust["Sales"].cumsum() / sorted_cust["Sales"].sum() * 100
    idx = (sorted_cust["CumSalesPct"] >= 80).idxmax()
    pct = (idx + 1) / len(sorted_cust) * 100
    rows.append(_insight(
        "Customer", "Customer Concentration",
        f"Approximately {pct:.0f}% of customers ({idx + 1} accounts) drive 80% of revenue.",
        "Maintain balanced portfolio; prioritize at-risk high-value customers for retention.",
        "customer_concentration", "Info",
    ))

    rfm = rfm_clusters(compute_rfm(df), n_clusters)
    champions = (rfm["Cluster Label"] == "Champions").sum()
    at_risk = (rfm["Cluster Label"] == "At Risk").sum()
    rows.append(_insight(
        "Advanced (RFM)", "RFM Segmentation",
        f"K-Means identified {n_clusters} behavioral clusters: {champions} Champions, {at_risk} At Risk (high value, low recency).",
        "Launch targeted retention campaigns for At Risk and Champions segments.",
        "rfm_clusters", "Opportunity",
    ))

    return pd.DataFrame(rows)


PRIORITY_TAG = {
    "Critical": "tag-critical",
    "Warning": "tag-warning",
    "Opportunity": "tag-opportunity",
    "Info": "tag-info",
}


def _insight_expander_label(priority: str, title: str) -> str:
    return f"[{priority}] {title}"


def _chart_widget_id(group: str, title: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in f"{group}_{title}")


def render_insights_tab(df: pd.DataFrame, n_clusters: int = 4):
    from utils.insight_charts import render_insight_chart

    page_title("Executive Insights", "Expand each insight to view finding, recommendation, and chart")
    insights = generate_insights(df, n_clusters)

    filter_group = st.multiselect(
        "Filter by group",
        INSIGHT_GROUPS,
        default=INSIGHT_GROUPS,
        key="insights_filter_group",
    )

    st.caption("Click an insight to expand and view its supporting chart.")

    for group in INSIGHT_GROUPS:
        if group not in filter_group:
            continue
        group_rows = insights[insights["Group"] == group]
        if group_rows.empty:
            continue

        st.markdown(f'<p class="insights-group-title">{group}</p>', unsafe_allow_html=True)

        for _, row in group_rows.iterrows():
            widget_id = _chart_widget_id(row["Group"], row["Insight"])
            label = _insight_expander_label(row["Priority"], row["Insight"])

            with st.expander(label, expanded=False):
                tag_cls = PRIORITY_TAG.get(row["Priority"], "tag-info")
                st.markdown(
                    f'<span class="{tag_cls}">[{row["Priority"]}]</span> '
                    f'<strong>{row["Insight"]}</strong>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="insight-detail-text">'
                    f'<span class="insight-detail-label">Finding:</span> {row["Finding"]}<br>'
                    f'<span class="insight-detail-label">Recommendation:</span> {row["Recommendation"]}'
                    f'</p>',
                    unsafe_allow_html=True,
                )

                fig = render_insight_chart(row["ChartKey"], df, n_clusters)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key=f"insight_chart_{widget_id}")
                else:
                    st.info("Chart not available for current filter selection.")

