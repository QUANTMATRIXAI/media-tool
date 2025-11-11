import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import io
import json
from datetime import datetime, timedelta


st.set_page_config(page_title="EDA Insights", page_icon="📊", layout="wide")


# Initialize session state variables for UI interactions
def initialize_session_state():
    """Initialize all session state variables used throughout the application"""
    
    # Comment system - edit mode tracking
    if 'editing_comment_id' not in st.session_state:
        st.session_state.editing_comment_id = None
    
    # Comment system - edit text buffer
    if 'edit_text' not in st.session_state:
        st.session_state.edit_text = ""
    
    # Comment system - delete confirmation tracking
    if 'delete_confirm_id' not in st.session_state:
        st.session_state.delete_confirm_id = None
    
    # Comment system - comment input text
    if 'comment_text' not in st.session_state:
        st.session_state.comment_text = ""
    
    # Comment system - optional heading
    if 'comment_heading' not in st.session_state:
        st.session_state.comment_heading = ""
    
    # Comment system - use heading option
    if 'use_heading' not in st.session_state:
        st.session_state.use_heading = False
    
    # Comment system - use bullet points option
    if 'use_bullets' not in st.session_state:
        st.session_state.use_bullets = False
    
    # Comment system - success/error messages
    if 'comment_success_message' not in st.session_state:
        st.session_state.comment_success_message = None
    
    if 'comment_error_message' not in st.session_state:
        st.session_state.comment_error_message = None
    
    # Date range tracking
    if 'start_date' not in st.session_state:
        st.session_state.start_date = None
    
    if 'end_date' not in st.session_state:
        st.session_state.end_date = None
    
    # Metric selection tracking
    if 'selected_metrics' not in st.session_state:
        st.session_state.selected_metrics = []
    
    # Aggregation preferences
    if 'agg_preferences' not in st.session_state:
        st.session_state.agg_preferences = {}
    
    # Metric selection by category
    for key in ['sel_web_totals', 'sel_web_gender', 'sel_media_type', 'sel_web_outcome', 'sel_perf', 
                'sel_sales_gender', 'sel_sales_summary', 'sel_sales_segment', 'manual_selected_metrics']:
        if key not in st.session_state:
            st.session_state[key] = []


# Initialize session state at app startup
initialize_session_state()


def cleanup_comment_state():
    """Clean up comment-related session state after operations"""
    st.session_state.comment_text = ""
    st.session_state.comment_heading = ""
    st.session_state.use_heading = False
    st.session_state.use_bullets = False
    st.session_state.editing_comment_id = None
    st.session_state.edit_text = ""
    st.session_state.delete_confirm_id = None


def cleanup_message_state():
    """Clean up success/error message state"""
    st.session_state.comment_success_message = None
    st.session_state.comment_error_message = None


# Title
st.title("📊 EDA Insights")


# Import file manager for persistent storage
from file_manager import smart_file_uploader, show_file_management_panel, get_current_project

# Sidebar Navigation
st.sidebar.markdown("### 🏠 Navigation")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("📁 Projects", use_container_width=True, key="back_to_projects"):
        st.session_state['current_page'] = 'eda_projects'
        st.rerun()
with col2:
    if st.button("🏠 Home", use_container_width=True, key="back_to_home"):
        st.session_state['current_page'] = 'home'
        st.rerun()

st.sidebar.divider()

# Get current project (set from project selection page)
current_project = get_current_project('eda')

# File Upload Section
st.sidebar.header("📁 Upload Data Files")

# Show current project
if current_project:
    st.sidebar.success(f"📂 Project: **{current_project}**")
    
    # Use smart file uploader that saves files per user per project
    uploaded_file = smart_file_uploader(
        label="Upload your data file (CSV/Excel)",
        file_types=['csv', 'xlsx', 'xls'],
        module='eda',
        file_key='main_data',
        help="Upload a CSV or Excel file with your analytics data"
    )
    
    # Show saved files panel
    st.sidebar.divider()
    show_file_management_panel('eda')
else:
    uploaded_file = None
    st.sidebar.error("❌ No project selected")
    if st.sidebar.button("🔙 Back to Projects", use_container_width=True):
        st.session_state['current_page'] = 'eda_projects'
        st.rerun()

# Sample template download
st.sidebar.divider()
st.sidebar.markdown("**Need a template?**")
sample_data = pd.DataFrame({
    'Day': pd.date_range('2025-01-01', periods=10),
    'Impressions': [1000, 1200, 1100, 1300, 1400, 1250, 1350, 1450, 1500, 1600],
    'Link_Clicks': [50, 60, 55, 65, 70, 62, 67, 72, 75, 80],
    'Bounce rate': [0.72, 0.71, 0.73, 0.70, 0.69, 0.71, 0.70, 0.68, 0.67, 0.69],
    'Sessions': [500, 550, 520, 580, 600, 560, 590, 610, 630, 650]
})
csv_template = sample_data.to_csv(index=False)
st.sidebar.download_button(
    label="📥 Download Sample Template",
    data=csv_template,
    file_name="sample_template.csv",
    mime="text/csv",
    use_container_width=True
)


# Load data function
@st.cache_data(show_spinner=False)
def load_uploaded_data(_file):
    """Load data from uploaded file (underscore prefix to skip hashing)"""
    try:
        if _file.name.endswith('.csv'):
            df = pd.read_csv(_file)
        else:
            df = pd.read_excel(_file)

        # Try to find and convert date column
        date_cols = [col for col in df.columns if 'day' in col.lower() or 'date' in col.lower()]
        if date_cols:
            # Try to parse dates with flexible format handling
            try:
                # First try with infer_datetime_format for automatic detection
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], infer_datetime_format=True)
            except:
                try:
                    # Try with dayfirst=True for DD-MM-YYYY or DD/MM/YYYY formats
                    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], dayfirst=True)
                except:
                    try:
                        # Try mixed format as last resort
                        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], format='mixed', dayfirst=True)
                    except:
                        # If all else fails, let pandas infer
                        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            
            # Rename to 'Day' for consistency
            if date_cols[0] != 'Day':
                df = df.rename(columns={date_cols[0]: 'Day'})

        # Ensure numeric types for common metric patterns (handles commas/$)
        def coerce_numeric_columns(frame):
            if frame is None or frame.empty:
                return frame
            patterns = [
                lambda c: c.endswith('_sales'),
                lambda c: c.endswith('_quantity'),
                lambda c: c.startswith('Impressions_') or c.startswith('Link_Clicks_') or c.startswith('Amount_Spent_'),
                lambda c: c in ['Impressions_Total', 'Link_Clicks_Total', 'Amount_Spent_Total'],
            ]
            for col in frame.columns:
                try:
                    if any(p(col) for p in patterns):
                        frame[col] = pd.to_numeric(
                            frame[col].astype(str).str.replace(r'[^0-9.\-]', '', regex=True),
                            errors='coerce'
                        )
                except Exception:
                    # Keep original if coercion fails for a column
                    pass
            return frame

        df = coerce_numeric_columns(df)

        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


# Check if file is uploaded
if uploaded_file is not None:
    df = load_uploaded_data(uploaded_file)

    if df is not None:
        st.sidebar.success(f"✅ File loaded: {uploaded_file.name}")
        st.sidebar.info(f"📊 Rows: {len(df):,} | Columns: {len(df.columns)}")
        
        # Track current file to detect changes
        if 'current_file_name' not in st.session_state:
            st.session_state.current_file_name = None
        
        # Reset selections if file changed
        if st.session_state.current_file_name != uploaded_file.name:
            st.session_state.current_file_name = uploaded_file.name
            st.session_state.selected_metrics = []
            st.session_state.sidebar_selected_metrics = []
            # Reset date range
            st.session_state.start_date = None
            st.session_state.end_date = None
            # Reset category selections
            for key in ['sel_web_totals', 'sel_web_gender', 'sel_media_type', 'sel_web_outcome', 
                       'sel_perf', 'sel_sales_gender', 'sel_sales_summary', 'sel_sales_segment', 
                       'manual_selected_metrics']:
                st.session_state[key] = []
        
        # Set smart default metrics if none are selected
        if len(st.session_state.selected_metrics) == 0:
            # Get all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Try to find common metric names
            default_candidates = [
                'Impressions', 'Link_Clicks', 'Amount_Spent_USD', 'Bounce rate',
                'Impressions_Total', 'Link_Clicks_Total', 'Amount_Spent_Total',
                'Sessions', 'Clicks', 'Conversions', 'Revenue', 'Sales'
            ]
            
            # Select up to 4 metrics that exist in the file
            smart_defaults = []
            for candidate in default_candidates:
                if candidate in numeric_cols:
                    smart_defaults.append(candidate)
                    if len(smart_defaults) >= 4:
                        break
            
            # If no common names found, just take first 4 numeric columns
            if len(smart_defaults) == 0 and len(numeric_cols) > 0:
                smart_defaults = numeric_cols[:min(4, len(numeric_cols))]
            
            st.session_state.selected_metrics = smart_defaults
else:
    st.warning("⚠️ Please upload a data file to begin analysis")
    st.info("""
    **Expected file format:**
    - CSV or Excel file
    - Must contain a 'Day' or 'Date' column
    - Should include numeric columns for metrics (Impressions, Sessions, etc.)

    **Example columns:**
    - Day, Impressions, Link_Clicks, Bounce rate, Sessions, etc.
    - Or: Day, Impressions_female, Impressions_male, Amount_Spent_Total, etc.
    """)
    st.stop()


# Get actual data range
if 'Day' in df.columns:
    min_date_raw = pd.to_datetime(df['Day'].min())
    max_date_raw = pd.to_datetime(df['Day'].max())
else:
    st.error("❌ No 'Day' or 'Date' column found in the uploaded file")
    st.stop()


# Show data preview in sidebar
with st.sidebar.expander("👁️ Preview Data", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Showing first 10 of {len(df)} rows")


# Show available columns
with st.sidebar.expander("📋 Available Columns", expanded=False):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.write("**Numeric columns:**")
    for col in numeric_cols:
        st.text(f"• {col}")
    st.caption(f"Total: {len(numeric_cols)} numeric columns")


# Quick metric selector in sidebar
st.sidebar.divider()
st.sidebar.subheader("🎯 Quick Select Metrics")

# Initialize sidebar selection in session state
if 'sidebar_selected_metrics' not in st.session_state:
    st.session_state.sidebar_selected_metrics = []

# Get all numeric columns
all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Clean up sidebar selections
st.session_state.sidebar_selected_metrics = [m for m in st.session_state.sidebar_selected_metrics if m in all_numeric_cols]

sidebar_metrics = st.sidebar.multiselect(
    "Select metrics to analyze",
    options=all_numeric_cols,
    default=st.session_state.sidebar_selected_metrics,
    key="sidebar_metric_selector",
    help="Select metrics directly from here or use the configuration section below"
)

if st.sidebar.button("✅ Apply Selection", use_container_width=True, key="apply_sidebar_selection"):
    st.session_state.sidebar_selected_metrics = sidebar_metrics
    st.session_state.selected_metrics = list(dict.fromkeys(sidebar_metrics))
    st.success(f"✅ {len(sidebar_metrics)} metrics selected!")
    st.rerun()

if sidebar_metrics:
    st.sidebar.caption(f"Selected: {len(sidebar_metrics)} metrics")


# Helper function to format metric values
def format_metric_value(value, metric_name):
    """Format a metric value based on its characteristics"""
    # Currency columns
    if 'Amount' in metric_name or 'Spent' in metric_name or 'sales' in metric_name or 'Sales' in metric_name:
        return f"${value:,.2f}"
    # Percentage/rate columns
    elif 'rate' in metric_name or 'Rate' in metric_name or 'percentage' in metric_name or 'Percentage' in metric_name:
        return f"{value:.2%}"
    # Large numbers or whole numbers
    elif value >= 1000 or (value % 1 == 0 and value < 1000):
        return f"{value:,.0f}"
    # Default: 2 decimal places
    else:
        return f"{value:,.2f}"


# Configuration Section
st.sidebar.divider()
st.sidebar.header("⚙️ Configuration")


# User-configurable aggregation settings
st.sidebar.subheader("🔧 Aggregation Settings")


# Default aggregation method
default_agg_method = st.sidebar.selectbox(
    "Default Aggregation Method",
    ["Sum", "Mean", "Median", "Min", "Max"],
    index=0,
    help="Choose how to aggregate metrics when grouping by period"
)


# Option to customize per metric
use_custom_agg_config = st.sidebar.checkbox(
    "Customize per metric",
    value=False,
    help="Set different aggregation methods for each metric"
)


# Aggregation preferences are initialized in initialize_session_state()


# Aggregate data function
def aggregate_data(df, level, agg_preferences, default_method):
    df_agg = df.copy()

    if level == "Weekly":
        df_agg['Period'] = df_agg['Day'].dt.to_period('W').apply(lambda r: r.start_time)
    elif level == "Monthly":
        df_agg['Period'] = df_agg['Day'].dt.to_period('M').apply(lambda r: r.start_time)
    elif level == "Quarterly":
        df_agg['Period'] = df_agg['Day'].dt.to_period('Q').apply(lambda r: r.start_time)
    else:
        df_agg['Period'] = df_agg['Day']

    numeric_cols = df_agg.select_dtypes(include=[np.number]).columns

    # Build aggregation dictionary based on user preferences
    agg_dict = {}
    for col in numeric_cols:
        # Use custom preference if available, otherwise use default
        method = agg_preferences.get(col, default_method).lower()
        agg_dict[col] = method

    df_result = df_agg.groupby('Period').agg(agg_dict).reset_index()
    return df_result


# Custom aggregation configuration
if use_custom_agg_config:
    with st.sidebar.expander("⚙️ Configure Aggregation per Metric", expanded=True):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.markdown("**Set aggregation method for each metric:**")

        for col in numeric_cols[:10]:  # Show first 10 to avoid clutter
            st.session_state.agg_preferences[col] = st.selectbox(
                col,
                ["Sum", "Mean", "Median", "Min", "Max"],
                key=f"agg_pref_{col}",
                index=0
            )

        if len(numeric_cols) > 10:
            st.info(f"Showing first 10 of {len(numeric_cols)} metrics. Others will use default method.")


# Metric selection
st.sidebar.divider()
st.sidebar.subheader("📊 Select Metrics")


"""Dynamic categorization based on pipeline output column patterns.
Groups appear only if matching columns exist; app remains robust if some are missing."""

# Get all numeric columns from the dataframe
all_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Pattern-based buckets
categories_pattern = {
    "Meta - Impressions (by segment)": [c for c in df.columns if c.startswith('Impressions_')],
    "Meta - Link Clicks (by segment)": [c for c in df.columns if c.startswith('Link_Clicks_')],
    "Meta - Spend (by segment, USD)": [c for c in df.columns if c.startswith('Amount_Spent_')],
    "Meta - Totals": [c for c in df.columns if c in ['Impressions_Total', 'Link_Clicks_Total', 'Amount_Spent_Total', 'Landing page views']],
    "Meta - Media Type": [c for c in df.columns if any(media in c for media in ['Image_', 'Video_', 'Image and video_'])],
    "Website - Outcome by Objective": [c for c in df.columns if any(obj in c for obj in ['APP_INSTALLS_', 'LINK_CLICKS_', 'OUTCOME_ENGAGEMENT_', 'OUTCOME_LEADS_', 'OUTCOME_SALES_'])],
    "Website - Traffic & Behavior": [c for c in df.columns if c in ['Bounce rate', 'Sessions', 'Sessions with cart additions', 'New customers', 'Customers']],
    "Website - Checkout Funnel": [c for c in df.columns if c in ['Sessions that reached checkout', 'Sessions that completed checkout', 'Sessions that reached and completed checkout']],
    "Sales - Segment": [c for c in df.columns if c in ['sales_new','sales_returning','quantity_new','quantity_returning']],
    "Shopify - Sales ($) by Gender": [c for c in df.columns if c.endswith('_sales')],
    "Shopify - Quantity by Gender": [c for c in df.columns if c.endswith('_quantity')],
    "Sales - Orders Summary": [c for c in df.columns if c in ['Quantity ordered', 'Gross sales', 'Net items sold']]
}

# Build available metrics by category (only non-empty lists, numeric only)
available_by_category = {}
categorized_set = set()
for cat, cols in categories_pattern.items():
    cols_num = [c for c in cols if c in all_columns]
    if cols_num:
        available_by_category[cat] = sorted(cols_num)
        categorized_set.update(cols_num)

# Others: numeric columns not captured above
other_metrics = sorted([c for c in all_columns if c not in categorized_set])
if other_metrics:
    available_by_category["Others"] = other_metrics

# Map higher-level dimensions to categories
dimension_to_categories = {
    "Meta Advertising": [
        "Meta - Impressions (by segment)",
        "Meta - Link Clicks (by segment)",
        "Meta - Spend (by segment, USD)",
        "Meta - Totals",
        "Meta - Media Type",
    ],
    "Website Performance": [
        "Website - Outcome by Objective",
        "Website - Traffic & Behavior",
        "Website - Checkout Funnel",
    ],
    "Sales / E-commerce": [
        "Sales - Segment",
        "Shopify - Sales ($) by Gender",
        "Shopify - Quantity by Gender",
        "Sales - Orders Summary",
    ],
}

# Keep only dimensions that have at least one available category present
available_dimensions = {}
for dim, cats in dimension_to_categories.items():
    present_cats = [c for c in cats if c in available_by_category]
    if present_cats:
        available_dimensions[dim] = present_cats


st.divider()
st.subheader("⚙️ Configuration - Metric Picker")

with st.expander("Select metrics to analyze", expanded=True):
    # Selected metrics are initialized in initialize_session_state()

    # Build Website split into Totals and Gender (Meta only), and keep Performance separate
    web_totals = sorted(available_by_category.get("Meta - Totals", []))
    web_gender = []
    for cat in [
        "Meta - Impressions (by segment)",
        "Meta - Link Clicks (by segment)",
        "Meta - Spend (by segment, USD)",
    ]:
        web_gender.extend(available_by_category.get(cat, []))
    web_gender = sorted(dict.fromkeys(web_gender))

    # Media Type
    media_type = sorted(available_by_category.get("Meta - Media Type", []))
    
    # Website Outcome by Objective
    web_outcome = sorted(available_by_category.get("Website - Outcome by Objective", []))
    
    # Performance stays as one list
    performance_options = []
    for cat in ["Website - Traffic & Behavior", "Website - Checkout Funnel"]:
        performance_options.extend(available_by_category.get(cat, []))
    performance_options = sorted(dict.fromkeys(performance_options))

    # Sales split into By Gender, Segment (New/Returning), and Summary
    sales_by_gender = []
    for cat in ["Shopify - Sales ($) by Gender", "Shopify - Quantity by Gender"]:
        sales_by_gender.extend(available_by_category.get(cat, []))
    sales_by_gender = sorted(dict.fromkeys(sales_by_gender))
    sales_summary = sorted(available_by_category.get("Sales - Orders Summary", []))
    sales_segment = sorted(available_by_category.get("Sales - Segment", []))

    # Persist selections per sub-dimension (initialized in initialize_session_state())
    # Clean up selections to only include valid metrics
    st.session_state.sel_web_totals = [m for m in st.session_state.sel_web_totals if m in web_totals]
    st.session_state.sel_web_gender = [m for m in st.session_state.sel_web_gender if m in web_gender]
    st.session_state.sel_media_type = [m for m in st.session_state.sel_media_type if m in media_type]
    st.session_state.sel_web_outcome = [m for m in st.session_state.sel_web_outcome if m in web_outcome]
    st.session_state.sel_perf = [m for m in st.session_state.sel_perf if m in performance_options]
    st.session_state.sel_sales_gender = [m for m in st.session_state.sel_sales_gender if m in sales_by_gender]
    st.session_state.sel_sales_summary = [m for m in st.session_state.sel_sales_summary if m in sales_summary]
    st.session_state.sel_sales_segment = [m for m in st.session_state.sel_sales_segment if m in sales_segment]

    # Layout in two rows of three columns (Row1: Website Totals | Website Gender | Media Type)
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        st.markdown("### Website - Totals")
        if web_totals:
            sel_web_totals = st.multiselect(
                "Totals",
                options=web_totals,
                default=st.session_state.sel_web_totals,
                key="cfg_sel_web_totals"
            )
            st.caption(f"Selected: {len(sel_web_totals)}")
        else:
            st.info("No metrics in this category")
            sel_web_totals = []

    with row1_col2:
        st.markdown("### Website - Gender")
        if web_gender:
            sel_web_gender = st.multiselect(
                "By Gender/Segment",
                options=web_gender,
                default=st.session_state.sel_web_gender,
                key="cfg_sel_web_gender"
            )
            st.caption(f"Selected: {len(sel_web_gender)}")
        else:
            st.info("No metrics in this category")
            sel_web_gender = []

    with row1_col3:
        st.markdown("### Meta - Media Type")
        if media_type:
            sel_media_type = st.multiselect(
                "By Media Type",
                options=media_type,
                default=st.session_state.sel_media_type,
                key="cfg_sel_media_type"
            )
            st.caption(f"Selected: {len(sel_media_type)}")
        else:
            st.info("No metrics in this category")
            sel_media_type = []

    # Row 2: Website Outcome | Performance | Sales - By Gender
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        st.markdown("### Website - Outcome")
        if web_outcome:
            sel_web_outcome = st.multiselect(
                "By Objective",
                options=web_outcome,
                default=st.session_state.sel_web_outcome,
                key="cfg_sel_web_outcome"
            )
            st.caption(f"Selected: {len(sel_web_outcome)}")
        else:
            st.info("No metrics in this category")
            sel_web_outcome = []

    with row2_col2:
        st.markdown("### Performance")
        if performance_options:
            sel_perf = st.multiselect(
                "Performance metrics",
                options=performance_options,
                default=st.session_state.sel_perf,
                key="cfg_sel_performance"
            )
            st.caption(f"Selected: {len(sel_perf)}")
        else:
            st.info("No metrics in this category")
            sel_perf = []

    with row2_col3:
        st.markdown("### Sales - By Gender")
        if sales_by_gender:
            sel_sales_gender = st.multiselect(
                "Sales by Gender",
                options=sales_by_gender,
                default=st.session_state.sel_sales_gender,
                key="cfg_sel_sales_gender"
            )
            st.caption(f"Selected: {len(sel_sales_gender)}")
        else:
            st.info("No metrics in this category")
            sel_sales_gender = []

    # Row 3: Sales - Summary | Sales - Segment | (empty)
    row3_col1, row3_col2, row3_col3 = st.columns(3)
    with row3_col1:
        st.markdown("### Sales - Summary")
        if sales_summary:
            sel_sales_summary = st.multiselect(
                "Sales summary",
                options=sales_summary,
                default=st.session_state.sel_sales_summary,
                key="cfg_sel_sales_summary"
            )
            st.caption(f"Selected: {len(sel_sales_summary)}")
        else:
            st.info("No metrics in this category")
            sel_sales_summary = []

    with row3_col2:
        st.markdown("### Sales - Segment")
        if sales_segment:
            sel_sales_segment = st.multiselect(
                "Sales segment (New/Returning)",
                options=sales_segment,
                default=st.session_state.sel_sales_segment,
                key="cfg_sel_sales_segment"
            )
            st.caption(f"Selected: {len(sel_sales_segment)}")
        else:
            st.info("No metrics in this category")
            sel_sales_segment = []
        if sel_sales_segment:
            st.write("\n".join([f"• {m}" for m in sel_sales_segment[:10]]))
            if len(sel_sales_segment) > 10:
                st.caption(f"(+{len(sel_sales_segment)-10} more)")

    # Apply button to avoid mid-change reruns clearing selections
    apply_cols = st.columns([1,1,1])
    with apply_cols[1]:
        if st.button("Apply selections", use_container_width=True, key="apply_metric_selection"):
            st.session_state.sel_web_totals = sel_web_totals
            st.session_state.sel_web_gender = sel_web_gender
            st.session_state.sel_media_type = sel_media_type
            st.session_state.sel_web_outcome = sel_web_outcome
            st.session_state.sel_perf = sel_perf
            st.session_state.sel_sales_gender = sel_sales_gender
            st.session_state.sel_sales_summary = sel_sales_summary
            st.session_state.sel_sales_segment = sel_sales_segment

            st.session_state.selected_metrics = (
                st.session_state.sel_web_totals +
                st.session_state.sel_web_gender +
                st.session_state.sel_media_type +
                st.session_state.sel_web_outcome +
                st.session_state.sel_perf +
                st.session_state.sel_sales_gender +
                st.session_state.sel_sales_summary +
                st.session_state.sel_sales_segment
            )
            # Deduplicate while preserving order
            st.session_state.selected_metrics = list(dict.fromkeys(st.session_state.selected_metrics))

# Keep a local alias for convenience with existing code below
selected_metrics = st.session_state.selected_metrics

# Check if there are uncategorized columns or if no categories were found
all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorized_cols = set()
for cat_list in available_by_category.values():
    categorized_cols.update(cat_list)
uncategorized_cols = sorted([c for c in all_numeric_cols if c not in categorized_cols])

# Sidebar: Uncategorized/Manual metric selector
if uncategorized_cols or len(available_by_category) == 0:
    st.sidebar.divider()
    st.sidebar.subheader("📋 Additional Metrics")
    
    if len(available_by_category) == 0:
        st.sidebar.info("No predefined categories found. Select metrics manually below.")
        manual_selection_cols = all_numeric_cols
    else:
        st.sidebar.info(f"Found {len(uncategorized_cols)} uncategorized columns.")
        manual_selection_cols = uncategorized_cols
    
    # Initialize manual selection in session state
    if 'manual_selected_metrics' not in st.session_state:
        st.session_state.manual_selected_metrics = []
    
    # Clean up manual selections
    st.session_state.manual_selected_metrics = [m for m in st.session_state.manual_selected_metrics if m in manual_selection_cols]
    
    manual_metrics = st.sidebar.multiselect(
        "Select additional metrics",
        options=manual_selection_cols,
        default=st.session_state.manual_selected_metrics,
        key="manual_metric_selector",
        help="These metrics were not categorized automatically"
    )
    
    if st.sidebar.button("Apply Manual Selection", use_container_width=True):
        st.session_state.manual_selected_metrics = manual_metrics
        # Add to selected metrics
        st.session_state.selected_metrics = list(dict.fromkeys(st.session_state.selected_metrics + manual_metrics))
        st.rerun()
    
    if manual_metrics:
        st.sidebar.caption(f"Selected: {len(manual_metrics)} additional metrics")

# Update selected_metrics to include manual selections
selected_metrics = list(dict.fromkeys(st.session_state.selected_metrics + st.session_state.get('manual_selected_metrics', [])))

# Sidebar summary of current selection
st.sidebar.divider()
with st.sidebar.expander("✅ All Selected Metrics", expanded=False):
    st.caption(f"Total Selected: {len(selected_metrics)}")
    if selected_metrics:
        st.write("\n".join([f"• {m}" for m in selected_metrics[:25]]))
        if len(selected_metrics) > 25:
            st.caption(f"(+{len(selected_metrics)-25} more)")


# Show info in sidebar
st.sidebar.divider()
st.sidebar.info(f"📅 Data: {min_date_raw.strftime('%Y-%m-%d')} to {max_date_raw.strftime('%Y-%m-%d')}")

# Initialize date range from data if not already set
if st.session_state.start_date is None or st.session_state.start_date < min_date_raw.date() or st.session_state.start_date > max_date_raw.date():
    st.session_state.start_date = min_date_raw.date()
if st.session_state.end_date is None or st.session_state.end_date < min_date_raw.date() or st.session_state.end_date > max_date_raw.date():
    st.session_state.end_date = max_date_raw.date()

# Default aggregation level for Overview tab
if 'agg_level' not in st.session_state:
    st.session_state.agg_level = 'Daily'

# Aggregate and filter data for Overview tab
df_agg = aggregate_data(df, st.session_state.get('agg_level', 'Daily'), st.session_state.agg_preferences, default_agg_method)
df_filtered = df_agg[
    (df_agg['Period'].dt.date >= st.session_state.start_date) & 
    (df_agg['Period'].dt.date <= st.session_state.end_date)
].copy()


if not selected_metrics:
    st.warning("⚠️ Please select at least one metric from the sidebar")
else:
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Overview", 
        "Explore",
        "Period Comparison",
        "Report",
        "Pivot Table",
        "Correlation Analysis",
        "Cardinality View",
        "Clustering"
    ])

    # Tab 1 - Overview (Data Table Only)
    with tab1:
        st.subheader("📊 Data Overview")
        st.markdown("View your filtered data in table format")
        
        # Data table
        display_cols = ['Period'] + [m for m in selected_metrics if m in df_filtered.columns]

        # Format columns appropriately based on their characteristics
        format_dict = {}
        for col in display_cols:
            if col == 'Period':
                continue
            # Currency columns
            elif 'Amount' in col or 'Spent' in col or 'sales' in col or 'Sales' in col:
                format_dict[col] = "${:,.2f}"
            # Percentage/rate columns
            elif 'rate' in col or 'Rate' in col or 'percentage' in col or 'Percentage' in col:
                format_dict[col] = "{:.2%}"
            # Large numbers or whole numbers
            else:
                format_dict[col] = "{:,.2f}"

        # Display styled dataframe
        if display_cols:
            styled_df = df_filtered[display_cols].style.format(format_dict)
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Download button
            csv_data = df_filtered[display_cols].to_csv(index=False)
            st.download_button(
                label="📥 Download Data",
                data=csv_data,
                file_name=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("No data to display")
    
    # Tab 2 - Explore (Charts and Analysis)
    with tab2:
        # Data Filters at the top
        with st.expander("🔍 Advanced Data Filters", expanded=False):
            # Get categorical columns (non-numeric, non-date)
            categorical_cols = [col for col in df_filtered.columns 
                              if col != 'Day' and df_filtered[col].dtype == 'object']
            
            if categorical_cols:
                # Filter controls
                col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
                with col_ctrl1:
                    st.markdown("**📊 Filter by categorical columns**")
                with col_ctrl2:
                    if st.button("✅ Select All", key="select_all_explore", use_container_width=True):
                        for col in categorical_cols[:6]:
                            st.session_state[f"filter_explore_{col}"] = list(df_filtered[col].dropna().unique())
                        st.rerun()
                with col_ctrl3:
                    if st.button("🗑️ Clear All", key="clear_all_explore", use_container_width=True):
                        for col in categorical_cols[:6]:
                            if f"filter_explore_{col}" in st.session_state:
                                st.session_state[f"filter_explore_{col}"] = []
                        st.rerun()
                
                st.divider()
                
                # Filters in grid
                filter_cols = st.columns(min(3, len(categorical_cols)))
                active_filters_explore = {}
                
                for idx, col in enumerate(categorical_cols[:6]):  # Limit to 6 filters
                    with filter_cols[idx % 3]:
                        unique_vals = df_filtered[col].dropna().unique()
                        
                        # Sort options
                        try:
                            unique_vals_sorted = sorted(unique_vals)
                        except:
                            unique_vals_sorted = list(unique_vals)
                        
                        if len(unique_vals_sorted) > 0 and len(unique_vals_sorted) <= 100:
                            # Show count
                            st.caption(f"**{col}** ({len(unique_vals_sorted)} options)")
                            
                            selected_vals = st.multiselect(
                                f"{col}",
                                options=unique_vals_sorted,
                                key=f"filter_explore_{col}",
                                label_visibility="collapsed"
                            )
                            if selected_vals:
                                active_filters_explore[col] = selected_vals
                
                # Apply filters
                if active_filters_explore:
                    for col, vals in active_filters_explore.items():
                        df_filtered = df_filtered[df_filtered[col].isin(vals)]
                    
                    # Show filter summary
                    st.success(f"✅ Active filters: {len(active_filters_explore)}")
                    filter_summary = " | ".join([f"**{col}**: {len(vals)} selected" for col, vals in active_filters_explore.items()])
                    st.caption(filter_summary)
                    st.metric("Filtered Rows", f"{len(df_filtered):,}", delta=f"{len(df_filtered) - len(df):,}")
            else:
                st.info("No categorical columns found for filtering")
        
        # Create two columns: Controls on left, Chart on right
        controls_col, chart_col = st.columns([1, 2.5])
        
        with controls_col:
            st.subheader("⚙️ Controls")
            
            # Date Range
            st.markdown("**📅 Date Range**")
            start_date_explore = st.date_input(
                "Start Date",
                value=st.session_state.start_date,
                min_value=min_date_raw.date(),
                max_value=max_date_raw.date(),
                key="start_date_explore"
            )
            st.session_state.start_date = start_date_explore
            
            end_date_explore = st.date_input(
                "End Date",
                value=st.session_state.end_date,
                min_value=min_date_raw.date(),
                max_value=max_date_raw.date(),
                key="end_date_explore"
            )
            st.session_state.end_date = end_date_explore
            
            # Quick date buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("30d", use_container_width=True, key="30d_explore"):
                    st.session_state.start_date = (max_date_raw - pd.Timedelta(days=30)).date()
                    st.session_state.end_date = max_date_raw.date()
                    st.rerun()
            with btn_col2:
                if st.button("90d", use_container_width=True, key="90d_explore"):
                    st.session_state.start_date = (max_date_raw - pd.Timedelta(days=90)).date()
                    st.session_state.end_date = max_date_raw.date()
                    st.rerun()
            with btn_col3:
                if st.button("All", use_container_width=True, key="all_explore"):
                    st.session_state.start_date = min_date_raw.date()
                    st.session_state.end_date = max_date_raw.date()
                    st.rerun()
            
            st.divider()
            
            # Aggregation
            st.markdown("**📊 Aggregation**")
            agg_level = st.selectbox(
                "Level",
                ["Daily", "Weekly", "Monthly", "Quarterly"],
                index=["Daily", "Weekly", "Monthly", "Quarterly"].index(st.session_state.get('agg_level', 'Daily')),
                key="agg_level_explore",
                label_visibility="collapsed"
            )
            st.session_state.agg_level = agg_level
            
            st.divider()
            
            # Chart Type
            st.markdown("**📈 Chart Type**")
            chart_type = st.selectbox(
                "Type",
                ["Line Chart", "Bar Chart", "Area Chart", "Pie Chart"],
                key="chart_type_explore",
                label_visibility="collapsed"
            )
            
            # Line Chart Options (only for Line Chart)
            line_chart_option = "Standard"
            primary_metrics = []
            secondary_metrics = []
            
            if chart_type == "Line Chart":
                st.divider()
                st.markdown("**⚙️ Line Chart Options**")
                line_chart_option = st.radio(
                    "Style",
                    ["Standard", "Dual Axis", "Normalized", "Separate Subplots"],
                    key="line_chart_option",
                    label_visibility="collapsed"
                )
                
                # Dual axis configuration (only for Dual Axis option)
                if line_chart_option == "Dual Axis" and len(selected_metrics) > 0:
                    st.divider()
                    st.markdown("**🔧 Dual Axis Setup**")
                    st.markdown("**Left Axis**")
                    for metric in selected_metrics:
                        if st.checkbox(f"{metric}", value=True, key=f"primary_explore_{metric}"):
                            primary_metrics.append(metric)
                    
                    if len(selected_metrics) > 1:
                        st.markdown("**Right Axis**")
                        remaining = [m for m in selected_metrics if m not in primary_metrics]
                        for metric in remaining:
                            if st.checkbox(f"{metric}", value=True, key=f"secondary_explore_{metric}"):
                                secondary_metrics.append(metric)
            
            st.divider()
        
        
        # Re-aggregate and filter data based on explore tab controls
        df_agg_explore = aggregate_data(df, agg_level, st.session_state.agg_preferences, default_agg_method)
        
        start_date = st.session_state.start_date
        end_date = st.session_state.end_date
        
        if start_date > end_date:
            st.error("⚠️ Start date must be before end date!")
            df_filtered_explore = df_agg_explore
        else:
            df_filtered_explore = df_agg_explore[
                (df_agg_explore['Period'].dt.date >= start_date) & 
                (df_agg_explore['Period'].dt.date <= end_date)
            ].copy()
            
            if len(df_filtered_explore) == 0:
                st.warning("⚠️ No data in selected range")
        
        with chart_col:
            st.subheader("📊 Visualization")
            
            # Show data info
            if len(df_filtered_explore) > 0:
                st.caption(f"Showing {len(df_filtered_explore)} {agg_level.lower()} periods from {start_date} to {end_date}")
            
            # Summary metrics (compact)
            metric_cols = st.columns(min(len(selected_metrics), 4))
            for idx, metric in enumerate(selected_metrics[:4]):  # Show max 4 metrics
                with metric_cols[idx]:
                    if metric in df_filtered_explore.columns:
                        agg_method = st.session_state.agg_preferences.get(metric, default_agg_method).lower()
                        if agg_method == 'sum':
                            total = df_filtered_explore[metric].sum()
                        elif agg_method == 'mean':
                            total = df_filtered_explore[metric].mean()
                        else:
                            total = df_filtered_explore[metric].sum()
                        value_str = format_metric_value(total, metric)
                        display_name = metric.replace('_', ' ')[:15] + '...' if len(metric) > 15 else metric.replace('_', ' ')
                        st.metric(label=display_name, value=value_str)
            
            if len(selected_metrics) > 4:
                st.caption(f"+ {len(selected_metrics) - 4} more metrics")
            
            st.divider()
            
            # Create visualization based on chart type
            if chart_type == "Line Chart":
                if line_chart_option == "Standard":
                    fig = go.Figure()
                    for metric in selected_metrics:
                        if metric in df_filtered_explore.columns:
                            fig.add_trace(go.Scatter(
                                x=df_filtered_explore['Period'],
                                y=df_filtered_explore[metric],
                                name=metric,
                                mode='lines+markers',
                                line=dict(width=2),
                                marker=dict(size=6)
                            ))
                    fig.update_layout(
                        title=f"Metrics Over Time ({agg_level})",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        hovermode='x unified',
                        height=600,
                        plot_bgcolor='#F5F5F5',
                        paper_bgcolor='#FFFFFF'
                    )
                
                elif line_chart_option == "Dual Axis":
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add primary axis metrics (left)
                    for metric in primary_metrics:
                        if metric in df_filtered_explore.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_filtered_explore['Period'],
                                    y=df_filtered_explore[metric],
                                    name=f"{metric} (Left)",
                                    mode='lines+markers',
                                    line=dict(width=2),
                                    marker=dict(size=6)
                                ),
                                secondary_y=False
                            )
                    
                    # Add secondary axis metrics (right)
                    for metric in secondary_metrics:
                        if metric in df_filtered_explore.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_filtered_explore['Period'],
                                    y=df_filtered_explore[metric],
                                    name=f"{metric} (Right)",
                                    mode='lines+markers',
                                    line=dict(width=2, dash='dash'),
                                    marker=dict(size=6, symbol='diamond')
                                ),
                                secondary_y=True
                            )
                    
                    # Build axis titles
                    left_title = ", ".join(primary_metrics) if primary_metrics else "Left Axis"
                    right_title = ", ".join(secondary_metrics) if secondary_metrics else "Right Axis"
                    
                    fig.update_layout(
                        title=f"Dual Axis Chart ({agg_level})",
                        hovermode='x unified',
                        height=600,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        plot_bgcolor='#F5F5F5',
                        paper_bgcolor='#FFFFFF'
                    )
                    fig.update_xaxes(title_text="Date")
                    fig.update_yaxes(title_text=left_title, secondary_y=False, showgrid=True)
                    fig.update_yaxes(title_text=right_title, secondary_y=True, showgrid=False)
                
                elif line_chart_option == "Normalized":
                    fig = go.Figure()
                    
                    for metric in selected_metrics:
                        if metric in df_filtered_explore.columns:
                            values = df_filtered_explore[metric].values
                            min_val = values.min()
                            max_val = values.max()
                            normalized = (values - min_val) / (max_val - min_val) if max_val > min_val else values
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=df_filtered_explore['Period'],
                                    y=normalized,
                                    name=metric,
                                    mode='lines+markers',
                                    line=dict(width=2),
                                    marker=dict(size=6),
                                    hovertemplate=f'<b>{metric}</b><br>Normalized: %{{y:.3f}}<br>Actual: %{{customdata:,.2f}}<extra></extra>',
                                    customdata=values
                                )
                            )
                    
                    fig.update_layout(
                        title=f"Normalized Metrics (Min-Max Scaling) - {agg_level}",
                        xaxis_title="Date",
                        yaxis_title="Normalized Value (0-1)",
                        hovermode='x unified',
                        height=600,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        plot_bgcolor='#F5F5F5',
                        paper_bgcolor='#FFFFFF'
                    )
                
                else:  # Separate Subplots
                    # Limit to max 800px height with scrolling
                    subplot_height = min(300 * len(selected_metrics), 800)
                    
                    fig = make_subplots(
                        rows=len(selected_metrics),
                        cols=1,
                        subplot_titles=selected_metrics,
                        vertical_spacing=0.08
                    )
                    
                    for idx, metric in enumerate(selected_metrics, 1):
                        if metric in df_filtered_explore.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_filtered_explore['Period'],
                                    y=df_filtered_explore[metric],
                                    name=metric,
                                    mode='lines+markers',
                                    line=dict(width=2),
                                    marker=dict(size=6),
                                    showlegend=False
                                ),
                                row=idx,
                                col=1
                            )
                            fig.update_yaxes(title_text=metric, row=idx, col=1)
                    
                    fig.update_layout(
                        title=f"Metrics Over Time ({agg_level})",
                        height=300 * len(selected_metrics),  # Full height for proper rendering
                        showlegend=False,
                        plot_bgcolor='#F5F5F5',
                        paper_bgcolor='#FFFFFF'
                    )
                    fig.update_xaxes(title_text="Date", row=len(selected_metrics), col=1)
                    
                    # Display with scrollable container
                    st.markdown(
                        f'<div style="height: {subplot_height}px; overflow-y: auto; border: 1px solid #E0E0E0; border-radius: 8px;">',
                        unsafe_allow_html=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Skip the normal plotly_chart call at the end
                    fig = None
            
            elif chart_type == "Bar Chart":
                fig = go.Figure()
                for metric in selected_metrics:
                    if metric in df_filtered_explore.columns:
                        fig.add_trace(go.Bar(
                            x=df_filtered_explore['Period'],
                            y=df_filtered_explore[metric],
                            name=metric
                        ))
                fig.update_layout(
                    title=f"Metrics Comparison ({agg_level})",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    barmode='group',
                    hovermode='x unified',
                    height=600,
                    plot_bgcolor='#F5F5F5',
                    paper_bgcolor='#FFFFFF'
                )
            
            elif chart_type == "Area Chart":
                fig = go.Figure()
                for metric in selected_metrics:
                    if metric in df_filtered_explore.columns:
                        fig.add_trace(go.Scatter(
                            x=df_filtered_explore['Period'],
                            y=df_filtered_explore[metric],
                            name=metric,
                            mode='lines',
                            fill='tonexty' if metric != selected_metrics[0] else 'tozeroy',
                            line=dict(width=0.5)
                        ))
                fig.update_layout(
                    title=f"Metrics Area Chart ({agg_level})",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode='x unified',
                    height=600,
                    plot_bgcolor='#F5F5F5',
                    paper_bgcolor='#FFFFFF'
                )
            
            elif chart_type == "Pie Chart":
                # Period selector for pie chart
                if len(df_filtered_explore) > 0:
                    st.markdown("**📅 Select Period for Pie Chart**")
                    
                    # Create list of periods
                    periods = df_filtered_explore['Period'].dt.strftime('%Y-%m-%d').tolist()
                    period_options = [f"{i+1}. {p}" for i, p in enumerate(periods)]
                    
                    selected_period = st.selectbox(
                        "Period",
                        options=period_options,
                        index=len(period_options) - 1,  # Default to latest
                        key="pie_period_selector",
                        label_visibility="collapsed"
                    )
                    
                    # Extract index from selection
                    selected_period_idx = period_options.index(selected_period)
                    
                    st.divider()
                    
                    # Get selected period data
                    selected_data = df_filtered_explore.iloc[selected_period_idx]
                    values = [selected_data[m] for m in selected_metrics if m in df_filtered_explore.columns]
                    labels = [m.replace('_', ' ') for m in selected_metrics if m in df_filtered_explore.columns]
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.3,
                        marker=dict(colors=['#FFBD59', '#41C185', '#458EE2', '#FFCF87', '#FFE7C2'])
                    )])
                    fig.update_layout(
                        title=f"Metrics Distribution - {selected_data['Period'].strftime('%Y-%m-%d')}",
                        height=600,
                        paper_bgcolor='#FFFFFF'
                    )
                else:
                    st.warning("No data available for pie chart")
                    fig = None
            

            
            # Display chart (if not already displayed in scrollable container)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            
            # Comments Section for Explore Tab
            st.divider()
            st.markdown("### 💬 Chart Analysis Comments")
            
            with st.expander("✍️ Add New Comment", expanded=False):
                comment_text_explore = st.text_area(
                    "Your observations about this chart",
                    placeholder="E.g., Notable spike in metrics during this period due to campaign launch...",
                    height=100,
                    key="explore_comment_text"
                )
                
                col_btn1, col_btn2 = st.columns([1, 3])
                with col_btn1:
                    if st.button("💾 Save Comment", use_container_width=True, key="save_explore_comment"):
                        if comment_text_explore and comment_text_explore.strip():
                            try:
                                import comment_manager
                                
                                # Create a summary of current chart analysis
                                chart_summary = pd.DataFrame({
                                    'Analysis Type': ['Explore - Chart Analysis'],
                                    'Chart Type': [chart_type],
                                    'Aggregation': [agg_level],
                                    'Date Range': [f"{start_date} to {end_date}"],
                                    'Metrics': [', '.join(selected_metrics[:5]) + (f' (+{len(selected_metrics)-5} more)' if len(selected_metrics) > 5 else '')]
                                })
                                
                                success, message = comment_manager.save_comment(
                                    comment_text=comment_text_explore,
                                    period1_start=start_date,
                                    period1_end=end_date,
                                    period2_start=start_date,
                                    period2_end=end_date,
                                    comparison_data=chart_summary,
                                    agg_method=agg_level,
                                    selected_metrics=selected_metrics
                                )
                                
                                if success:
                                    st.success("✅ Comment saved successfully!")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
                            except Exception as e:
                                st.error(f"❌ Error saving comment: {str(e)}")
                        else:
                            st.warning("⚠️ Please enter a comment before saving")
                
                with col_btn2:
                    if st.button("🗑️ Clear", use_container_width=True, key="clear_explore_comment"):
                        st.rerun()
            
            # Display saved comments
            try:
                import comment_manager
                comments_df = comment_manager.load_comments()
                
                if comments_df is not None and not comments_df.empty:
                    st.markdown("#### 📋 Recent Comments")
                    
                    # Show last 5 comments
                    for idx, row in comments_df.tail(5).iterrows():
                        with st.expander(f"💬 {row['timestamp'][:19]}", expanded=False):
                            st.markdown(f"**Comment:** {row['comment_text']}")
                            st.caption(f"📅 Period: {row['period1_start']} to {row['period1_end']} | Method: {row['agg_method']}")
                            
                            # Delete button
                            if st.button(f"🗑️ Delete", key=f"del_explore_{row['id']}"):
                                success, message = comment_manager.delete_comment(row['id'])
                                if success:
                                    st.success("✅ Comment deleted")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
            except Exception as e:
                st.info("💡 Comment system available - add your first comment above!")

    # Tab 3 - Period Comparison
    with tab3:
        st.subheader("📊 Period Comparison Analysis")

        st.markdown("Compare metrics between two time periods with custom aggregation")

        # Aggregation method selector
        agg_col1, agg_col2, agg_col3 = st.columns([2, 2, 5])
        with agg_col1:
            comparison_agg_method = st.selectbox(
                "Aggregation Method",
                ["Mean (Average)", "Sum (Total)", "Median", "Min (Minimum)", "Max (Maximum)"],
                key="comparison_agg_method",
                help="Choose how to aggregate values within each period"
            )

        with agg_col2:
            use_custom_agg = st.checkbox(
                "Custom per metric",
                value=False,
                help="Set different aggregation methods for each metric"
            )

        with agg_col3:
            if not use_custom_agg:
                st.info(f"📊 Using **{comparison_agg_method}** for all metrics")
            else:
                st.info("📊 Custom aggregation enabled - configure below")

        # Custom aggregation per metric
        metric_agg_methods = {}
        if use_custom_agg and selected_metrics:
            st.markdown("**Configure aggregation per metric:**")
            custom_cols = st.columns(min(3, len(selected_metrics)))
            for idx, metric in enumerate(selected_metrics):
                with custom_cols[idx % 3]:
                    display_name = metric.replace('_', ' ').replace('Amount Spent USD', 'Amount Spent (USD)')
                    metric_agg_methods[metric] = st.selectbox(
                        display_name,
                        ["Mean", "Sum", "Median", "Min", "Max"],
                        key=f"agg_{metric}",
                        index=0
                    )

        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns([2, 2, 2, 2])

        with comp_col1:
            period1_start = st.date_input(
                "Period 1 - Start",
                value=start_date,
                min_value=min_date_raw.date(),
                max_value=max_date_raw.date(),
                key="period1_start"
            )

        with comp_col2:
            period1_end = st.date_input(
                "Period 1 - End",
                value=start_date + pd.Timedelta(days=30) if start_date + pd.Timedelta(days=30) <= end_date else end_date,
                min_value=min_date_raw.date(),
                max_value=max_date_raw.date(),
                key="period1_end"
            )

        with comp_col3:
            period2_start = st.date_input(
                "Period 2 - Start",
                value=start_date + pd.Timedelta(days=31) if start_date + pd.Timedelta(days=31) <= end_date else start_date,
                min_value=min_date_raw.date(),
                max_value=max_date_raw.date(),
                key="period2_start"
            )

        with comp_col4:
            period2_end = st.date_input(
                "Period 2 - End",
                value=end_date,
                min_value=min_date_raw.date(),
                max_value=max_date_raw.date(),
                key="period2_end"
            )

        # Validate periods
        if period1_start <= period1_end and period2_start <= period2_end:
            # Filter data for both periods
            period1_data = df_agg[
                (df_agg['Period'].dt.date >= period1_start) & 
                (df_agg['Period'].dt.date <= period1_end)
            ]

            period2_data = df_agg[
                (df_agg['Period'].dt.date >= period2_start) & 
                (df_agg['Period'].dt.date <= period2_end)
            ]

            if len(period1_data) > 0 and len(period2_data) > 0:
                st.markdown(f"**Period 1:** {period1_start} to {period1_end} ({len(period1_data)} {agg_level.lower()} periods)")
                st.markdown(f"**Period 2:** {period2_start} to {period2_end} ({len(period2_data)} {agg_level.lower()} periods)")

                # Map aggregation method to pandas function
                agg_method_map = {
                    "Mean (Average)": "mean",
                    "Sum (Total)": "sum",
                    "Median": "median",
                    "Min (Minimum)": "min",
                    "Max (Maximum)": "max",
                    "Mean": "mean",
                    "Sum": "sum",
                    "Min": "min",
                    "Max": "max"
                }

                # Calculate metrics for selected aggregation method
                comparison_data = []
                
                # Calculate totals for percentage split
                # For Amount_Spent_ prefix columns
                p1_amount_spent_total = 0
                p2_amount_spent_total = 0
                p1_impressions_total = 0
                p2_impressions_total = 0
                p1_link_clicks_total = 0
                p2_link_clicks_total = 0
                
                # For Website Outcome columns (ending with specific suffixes)
                p1_outcome_usd_total = 0
                p2_outcome_usd_total = 0
                p1_outcome_impressions_total = 0
                p2_outcome_impressions_total = 0
                p1_outcome_clicks_total = 0
                p2_outcome_clicks_total = 0
                
                # First pass: calculate totals
                for metric in selected_metrics:
                    if metric in df_agg.columns:
                        # Determine aggregation method for this metric
                        if use_custom_agg and metric in metric_agg_methods:
                            selected_agg = agg_method_map[metric_agg_methods[metric]]
                        else:
                            selected_agg = agg_method_map[comparison_agg_method]
                        
                        # Apply selected aggregation method
                        if selected_agg == "mean":
                            p1_value = period1_data[metric].mean()
                            p2_value = period2_data[metric].mean()
                        elif selected_agg == "sum":
                            p1_value = period1_data[metric].sum()
                            p2_value = period2_data[metric].sum()
                        elif selected_agg == "median":
                            p1_value = period1_data[metric].median()
                            p2_value = period2_data[metric].median()
                        elif selected_agg == "min":
                            p1_value = period1_data[metric].min()
                            p2_value = period2_data[metric].min()
                        elif selected_agg == "max":
                            p1_value = period1_data[metric].max()
                            p2_value = period2_data[metric].max()
                        
                        # Accumulate totals for percentage calculation
                        # For prefix-based columns (Amount_Spent_, Impressions_, Link_Clicks_)
                        if metric.startswith('Amount_Spent_'):
                            p1_amount_spent_total += p1_value
                            p2_amount_spent_total += p2_value
                        elif metric.startswith('Impressions_'):
                            p1_impressions_total += p1_value
                            p2_impressions_total += p2_value
                        elif metric.startswith('Link_Clicks_'):
                            p1_link_clicks_total += p1_value
                            p2_link_clicks_total += p2_value
                        
                        # For suffix-based Website Outcome columns
                        if metric.endswith('(USD)'):
                            p1_outcome_usd_total += p1_value
                            p2_outcome_usd_total += p2_value
                        elif metric.endswith('Impressions'):
                            p1_outcome_impressions_total += p1_value
                            p2_outcome_impressions_total += p2_value
                        elif metric.endswith('Link clicks'):
                            p1_outcome_clicks_total += p1_value
                            p2_outcome_clicks_total += p2_value

                # Second pass: build comparison data with percentages
                for metric in selected_metrics:
                    if metric in df_agg.columns:
                        # Determine aggregation method for this metric
                        if use_custom_agg and metric in metric_agg_methods:
                            selected_agg = agg_method_map[metric_agg_methods[metric]]
                            agg_label = metric_agg_methods[metric]
                        else:
                            selected_agg = agg_method_map[comparison_agg_method]
                            agg_label = comparison_agg_method.split(" ")[0]

                        # Apply selected aggregation method
                        if selected_agg == "mean":
                            p1_value = period1_data[metric].mean()
                            p2_value = period2_data[metric].mean()
                        elif selected_agg == "sum":
                            p1_value = period1_data[metric].sum()
                            p2_value = period2_data[metric].sum()
                        elif selected_agg == "median":
                            p1_value = period1_data[metric].median()
                            p2_value = period2_data[metric].median()
                        elif selected_agg == "min":
                            p1_value = period1_data[metric].min()
                            p2_value = period2_data[metric].min()
                        elif selected_agg == "max":
                            p1_value = period1_data[metric].max()
                            p2_value = period2_data[metric].max()

                        # Calculate change
                        if p1_value != 0:
                            change_pct = ((p2_value - p1_value) / p1_value) * 100
                        else:
                            change_pct = 0 if p2_value == 0 else float('inf')

                        change_abs = p2_value - p1_value
                        
                        # Calculate percentage splits for specific metric types
                        p1_split_pct = None
                        p2_split_pct = None
                        
                        # For prefix-based columns (Amount_Spent_, Impressions_, Link_Clicks_)
                        if metric.startswith('Amount_Spent_') and p1_amount_spent_total > 0:
                            p1_split_pct = (p1_value / p1_amount_spent_total) * 100 if p1_amount_spent_total > 0 else 0
                            p2_split_pct = (p2_value / p2_amount_spent_total) * 100 if p2_amount_spent_total > 0 else 0
                        elif metric.startswith('Impressions_') and p1_impressions_total > 0:
                            p1_split_pct = (p1_value / p1_impressions_total) * 100 if p1_impressions_total > 0 else 0
                            p2_split_pct = (p2_value / p2_impressions_total) * 100 if p2_impressions_total > 0 else 0
                        elif metric.startswith('Link_Clicks_') and p1_link_clicks_total > 0:
                            p1_split_pct = (p1_value / p1_link_clicks_total) * 100 if p1_link_clicks_total > 0 else 0
                            p2_split_pct = (p2_value / p2_link_clicks_total) * 100 if p2_link_clicks_total > 0 else 0
                        
                        # For suffix-based Website Outcome columns
                        elif metric.endswith('(USD)') and p1_outcome_usd_total > 0:
                            p1_split_pct = (p1_value / p1_outcome_usd_total) * 100 if p1_outcome_usd_total > 0 else 0
                            p2_split_pct = (p2_value / p2_outcome_usd_total) * 100 if p2_outcome_usd_total > 0 else 0
                        elif metric.endswith('Impressions') and p1_outcome_impressions_total > 0:
                            p1_split_pct = (p1_value / p1_outcome_impressions_total) * 100 if p1_outcome_impressions_total > 0 else 0
                            p2_split_pct = (p2_value / p2_outcome_impressions_total) * 100 if p2_outcome_impressions_total > 0 else 0
                        elif metric.endswith('Link clicks') and p1_outcome_clicks_total > 0:
                            p1_split_pct = (p1_value / p1_outcome_clicks_total) * 100 if p1_outcome_clicks_total > 0 else 0
                            p2_split_pct = (p2_value / p2_outcome_clicks_total) * 100 if p2_outcome_clicks_total > 0 else 0

                        comparison_data.append({
                            'Metric': metric.replace('_', ' ').replace('Amount Spent USD', 'Amount Spent (USD)'),
                            'Agg Method': agg_label,
                            'Period 1': p1_value,
                            'Period 2': p2_value,
                            'Change': change_abs,
                            'Change %': change_pct,
                            'Period 1 Split %': p1_split_pct,
                            'Period 2 Split %': p2_split_pct
                        })

                # Create comparison dataframe
                comp_df = pd.DataFrame(comparison_data)

                # Display comparison table
                st.markdown("### 📈 Comparison Table")

                # Format the dataframe
                def format_value(val, metric):
                    if 'Amount Spent' in metric:
                        return f"${val:,.2f}"
                    elif 'Bounce rate' in metric:
                        return f"{val:.2%}"
                    elif any(x in metric for x in ['Impressions', 'Link Clicks', 'Sessions', 'Quantity']):
                        return f"{val:,.0f}"
                    else:
                        return f"{val:,.2f}"

                # Create formatted display
                display_data = []

                for _, row in comp_df.iterrows():
                    change_pct_str = f"{row['Change %']:+.2f}%" if row['Change %'] != float('inf') else "N/A"
                    
                    # Format Period 1 and Period 2 with split percentages if available
                    p1_display = format_value(row['Period 1'], row['Metric'])
                    p2_display = format_value(row['Period 2'], row['Metric'])
                    
                    # Add split percentage in brackets if available
                    if pd.notna(row['Period 1 Split %']) and row['Period 1 Split %'] is not None:
                        p1_display += f" ({row['Period 1 Split %']:.1f}%)"
                    if pd.notna(row['Period 2 Split %']) and row['Period 2 Split %'] is not None:
                        p2_display += f" ({row['Period 2 Split %']:.1f}%)"
                    
                    display_row = {
                        'Metric': row['Metric'],
                        'Period 1': p1_display,
                        'Period 2': p2_display,
                        'Change': format_value(row['Change'], row['Metric']),
                        'Change %': change_pct_str
                    }

                    # Add aggregation method column if custom aggregation is used
                    if use_custom_agg:
                        display_row['Agg Method'] = row['Agg Method']
                        # Reorder columns
                        display_row = {
                            'Metric': display_row['Metric'],
                            'Agg Method': display_row['Agg Method'],
                            'Period 1': display_row['Period 1'],
                            'Period 2': display_row['Period 2'],
                            'Change': display_row['Change'],
                            'Change %': display_row['Change %']
                        }

                    display_data.append(display_row)

                display_df = pd.DataFrame(display_data)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Visualization charts hidden - table is sufficient
                # Uncomment below if you want to show visual comparison charts
                
                # # Visualization
                # st.markdown("### 📊 Visual Comparison")

                # # Create bar chart comparison
                # fig_comp = go.Figure()

                # metrics_list = comp_df['Metric'].tolist()
                # period1_vals = comp_df['Period 1'].tolist()
                # period2_vals = comp_df['Period 2'].tolist()

                # # Add aggregation method to hover text if custom
                # hover_text_p1 = [f"{m}<br>Period 1: {format_value(v, m)}" + 
                #                 (f"<br>Method: {comp_df.iloc[i]['Agg Method']}" if use_custom_agg else "")
                #                 for i, (m, v) in enumerate(zip(metrics_list, period1_vals))]
                # hover_text_p2 = [f"{m}<br>Period 2: {format_value(v, m)}" + 
                #                 (f"<br>Method: {comp_df.iloc[i]['Agg Method']}" if use_custom_agg else "")
                #                 for i, (m, v) in enumerate(zip(metrics_list, period2_vals))]

                # fig_comp.add_trace(go.Bar(
                #     name='Period 1',
                #     x=metrics_list,
                #     y=period1_vals,
                #     marker_color='lightblue',
                #     hovertext=hover_text_p1,
                #     hoverinfo='text'
                # ))

                # fig_comp.add_trace(go.Bar(
                #     name='Period 2',
                #     x=metrics_list,
                #     y=period2_vals,
                #     marker_color='lightcoral',
                #     hovertext=hover_text_p2,
                #     hoverinfo='text'
                # ))

                # title_text = "Values Comparison" if use_custom_agg else f"{comparison_agg_method.split(' ')[0]} Values Comparison"

                # fig_comp.update_layout(
                #     title=title_text,
                #     xaxis_title="Metrics",
                #     yaxis_title="Value",
                #     barmode='group',
                #     height=500,
                #     hovermode='closest'
                # )

                # st.plotly_chart(fig_comp, use_container_width=True)

                # # Change percentage chart
                # st.markdown("### 📈 Percentage Change")

                # fig_change = go.Figure()

                # colors = ['green' if x > 0 else 'red' for x in comp_df['Change %']]

                # fig_change.add_trace(go.Bar(
                #     x=metrics_list,
                #     y=comp_df['Change %'].tolist(),
                #     marker_color=colors,
                #     text=[f"{x:+.1f}%" for x in comp_df['Change %']],
                #     textposition='outside'
                # ))

                # fig_change.update_layout(
                #     title="Percentage Change (Period 2 vs Period 1)",
                #     xaxis_title="Metrics",
                #     yaxis_title="Change %",
                #     height=400,
                #     showlegend=False
                # )

                # fig_change.add_hline(y=0, line_dash="dash", line_color="gray")

                # st.plotly_chart(fig_change, use_container_width=True)

                # Comment Section
                st.divider()
                st.subheader("💬 Add Comment")
                
                # Import comment manager
                import comment_manager
                
                # Comment text is initialized in initialize_session_state()
                
                # Formatting options
                format_col1, format_col2 = st.columns(2)
                with format_col1:
                    use_heading = st.checkbox(
                        "📝 Add heading",
                        value=st.session_state.use_heading,
                        key="use_heading_checkbox",
                        help="Add an optional heading to your comment"
                    )
                    st.session_state.use_heading = use_heading
                
                with format_col2:
                    use_bullets = st.checkbox(
                        "• Use bullet points",
                        value=st.session_state.use_bullets,
                        key="use_bullets_checkbox",
                        help="Format your comment with bullet points"
                    )
                    st.session_state.use_bullets = use_bullets
                
                # Optional heading input
                comment_heading = ""
                if use_heading:
                    comment_heading = st.text_input(
                        "Heading:",
                        value=st.session_state.comment_heading,
                        placeholder="Enter a heading for your comment...",
                        key="comment_heading_input"
                    )
                    st.session_state.comment_heading = comment_heading
                
                # Text area for comment input with dynamic placeholder
                if use_bullets:
                    placeholder_text = "Enter your comment using bullet points:\n• Point 1\n• Point 2\n• Point 3"
                else:
                    placeholder_text = "Enter your comment here..."
                
                comment_text = st.text_area(
                    "Comment:" if use_heading else "Add your insights and observations about this comparison:",
                    value=st.session_state.comment_text,
                    height=150,
                    placeholder=placeholder_text,
                    key="comment_input"
                )
                st.session_state.comment_text = comment_text
                
                # Character count
                char_count = len(comment_text)
                if use_heading:
                    char_count += len(comment_heading)
                st.caption(f"Characters: {char_count}")
                
                # Preview section
                if use_heading or use_bullets:
                    with st.expander("👁️ Preview", expanded=False):
                        if use_heading and comment_heading.strip():
                            st.markdown(f"### {comment_heading.strip()}")
                        if comment_text.strip():
                            if use_bullets:
                                # Format as bullet points if not already formatted
                                lines = comment_text.strip().split('\n')
                                formatted_lines = []
                                for line in lines:
                                    line = line.strip()
                                    if line:
                                        if not line.startswith('•') and not line.startswith('-') and not line.startswith('*'):
                                            formatted_lines.append(f"• {line}")
                                        else:
                                            formatted_lines.append(line)
                                st.markdown('\n'.join(formatted_lines))
                            else:
                                st.markdown(comment_text.strip())
                
                # Save button with validation
                save_col1, save_col2, save_col3 = st.columns([1, 2, 1])
                with save_col2:
                    # Disable if comment is empty, or if heading is enabled but both heading and text are empty
                    is_disabled = len(comment_text.strip()) == 0
                    if use_heading and len(comment_heading.strip()) == 0 and len(comment_text.strip()) == 0:
                        is_disabled = True
                    
                    save_button = st.button(
                        "💾 Save Comment",
                        use_container_width=True,
                        disabled=is_disabled,
                        key="save_comment_btn"
                    )
                
                # Display success/error messages from previous operations
                if st.session_state.comment_success_message:
                    st.success(st.session_state.comment_success_message)
                    cleanup_message_state()
                
                if st.session_state.comment_error_message:
                    st.error(st.session_state.comment_error_message)
                    cleanup_message_state()
                
                # Handle save button click
                if save_button:
                    with st.spinner("Saving comment..."):
                        # Validate that comparison data exists
                        if 'comp_df' not in locals() or comp_df is None or comp_df.empty:
                            st.session_state.comment_error_message = "❌ Cannot save comment: No comparison data available. Please ensure the comparison has been generated."
                            st.rerun()
                        else:
                            # Format the comment text with heading and bullet points
                            formatted_comment = ""
                            
                            # Add heading if enabled
                            if use_heading and comment_heading.strip():
                                formatted_comment += f"**{comment_heading.strip()}**\n\n"
                            
                            # Format bullet points if enabled
                            if use_bullets and comment_text.strip():
                                lines = comment_text.strip().split('\n')
                                formatted_lines = []
                                for line in lines:
                                    line = line.strip()
                                    if line:
                                        # Add bullet if not already present
                                        if not line.startswith('•') and not line.startswith('-') and not line.startswith('*'):
                                            formatted_lines.append(f"• {line}")
                                        else:
                                            # Normalize to use •
                                            if line.startswith('-') or line.startswith('*'):
                                                formatted_lines.append(f"• {line[1:].strip()}")
                                            else:
                                                formatted_lines.append(line)
                                formatted_comment += '\n'.join(formatted_lines)
                            else:
                                formatted_comment += comment_text.strip()
                            
                            # Determine the aggregation method string to save
                            if use_custom_agg:
                                agg_method_str = "Custom"
                            else:
                                agg_method_str = comparison_agg_method.split(" ")[0]
                            
                            # Call save_comment function with formatted text
                            success, message = comment_manager.save_comment(
                                comment_text=formatted_comment,
                                period1_start=period1_start,
                                period1_end=period1_end,
                                period2_start=period2_start,
                                period2_end=period2_end,
                                comparison_data=comp_df,
                                agg_method=agg_method_str,
                                selected_metrics=selected_metrics
                            )
                            
                            if success:
                                # Get current timestamp for display
                                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                st.session_state.comment_success_message = f"✅ {message} at {current_time}"
                                # Clear the comment state
                                cleanup_comment_state()
                                st.rerun()
                            else:
                                st.session_state.comment_error_message = f"❌ {message}"
                                st.rerun()

            else:
                st.warning("⚠️ No data available for one or both periods")
        else:
            st.error("⚠️ Invalid date ranges. Start date must be before end date for both periods.")

    # Tab 4 - Report
    with tab4:
        st.subheader("📋 Saved Comments Report")
        
        # Display success/error messages from previous operations
        if st.session_state.comment_success_message:
            st.success(st.session_state.comment_success_message)
            cleanup_message_state()
        
        if st.session_state.comment_error_message:
            st.error(st.session_state.comment_error_message)
            cleanup_message_state()
        
        # Import comment manager
        import comment_manager
        import json
        
        # Load all comments
        comments_df, load_error = comment_manager.load_comments()
        
        # Display load error if any
        if load_error:
            st.error(f"⚠️ Error loading comments: {load_error}")
        
        # Header with export button
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            st.markdown("View, edit, and manage all your saved comments")
        with header_col2:
            # Export functionality
            if not comments_df.empty:
                export_csv, export_error = comment_manager.export_comments()
                if export_error:
                    st.error(f"⚠️ {export_error}")
                elif export_csv:
                    current_date = datetime.now().strftime('%Y-%m-%d')
                    st.download_button(
                        label="📥 Export All",
                        data=export_csv,
                        file_name=f"comments_export_{current_date}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        st.divider()
        
        # Check if there are any comments
        if comments_df.empty:
            st.info("📝 No comments saved yet. Go to the Period Comparison tab to add your first comment!")
        else:
            # Session state for edit mode is initialized in initialize_session_state()
            
            # Sort comments by timestamp in descending order (newest first)
            comments_df = comments_df.sort_values('timestamp', ascending=False)
            
            st.caption(f"Total comments: {len(comments_df)}")
            
            # Display each comment
            for idx, row in comments_df.iterrows():
                comment_id = row['id']
                
                # Create a container for each comment
                with st.container():
                    # Comment header with metadata
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Extract heading from comment text if it exists (formatted as **Heading**)
                        comment_text = row['comment_text']
                        heading = None
                        
                        # Check if comment starts with bold markdown (heading)
                        if comment_text.startswith('**') and '**' in comment_text[2:]:
                            # Extract the heading between the first ** and the next **
                            end_pos = comment_text.index('**', 2)
                            heading = comment_text[2:end_pos].strip()
                        
                        # Display heading if found, otherwise show timestamp
                        if heading:
                            st.markdown(f"### 💬 {heading}")
                            st.caption(f"📅 {timestamp_str}")
                        else:
                            st.markdown(f"### 💬 Comment - {timestamp_str}")
                    
                    with col2:
                        # Edit button - only show if not in delete confirmation mode
                        if st.session_state.delete_confirm_id != comment_id:
                            if st.button("✏️ Edit", key=f"edit_btn_{comment_id}", use_container_width=True):
                                st.session_state.editing_comment_id = comment_id
                                st.session_state.edit_text = row['comment_text']
                                st.rerun()
                    
                    with col3:
                        # Delete button - only show if not in delete confirmation mode
                        if st.session_state.delete_confirm_id != comment_id:
                            if st.button("🗑️ Delete", key=f"delete_btn_{comment_id}", use_container_width=True):
                                st.session_state.delete_confirm_id = comment_id
                                st.rerun()
                    
                    # Delete confirmation dialog - show immediately after buttons
                    if st.session_state.delete_confirm_id == comment_id:
                        st.warning("⚠️ Are you sure you want to delete this comment? This action cannot be undone.")
                        
                        # Show a preview of what will be deleted
                        with st.expander("👁️ Preview comment to be deleted", expanded=False):
                            st.markdown(f"**Period 1:** {row['period1_start']} to {row['period1_end']}")
                            st.markdown(f"**Period 2:** {row['period2_start']} to {row['period2_end']}")
                            st.markdown(f"**Comment:** {row['comment_text'][:200]}{'...' if len(row['comment_text']) > 200 else ''}")
                        
                        confirm_col1, confirm_col2, confirm_col3 = st.columns([1, 1, 2])
                        with confirm_col1:
                            if st.button("✅ Yes, Delete", key=f"confirm_delete_{comment_id}", use_container_width=True):
                                success, message = comment_manager.delete_comment(comment_id)
                                if success:
                                    st.session_state.comment_success_message = f"✅ {message}"
                                    # Clean up delete confirmation state
                                    st.session_state.delete_confirm_id = None
                                    st.rerun()
                                else:
                                    st.session_state.comment_error_message = f"❌ {message}"
                                    st.rerun()
                        
                        with confirm_col2:
                            if st.button("❌ Cancel", key=f"cancel_delete_{comment_id}", use_container_width=True):
                                # Clean up delete confirmation state
                                st.session_state.delete_confirm_id = None
                                st.rerun()
                        
                        st.markdown("---")
                        st.markdown("")  # Add spacing
                        continue  # Skip showing the rest of the comment details
                    
                    # Metadata display
                    meta_col1, meta_col2, meta_col3 = st.columns(3)
                    with meta_col1:
                        st.markdown(f"**Period 1:** {row['period1_start']} to {row['period1_end']}")
                    with meta_col2:
                        st.markdown(f"**Period 2:** {row['period2_start']} to {row['period2_end']}")
                    with meta_col3:
                        st.markdown(f"**Aggregation:** {row['agg_method']}")
                    
                    # Display selected metrics
                    try:
                        if pd.isna(row['metrics']) or not row['metrics']:
                            st.markdown("**Metrics analyzed:** N/A")
                        else:
                            metrics_list = json.loads(row['metrics'])
                            if isinstance(metrics_list, list) and len(metrics_list) > 0:
                                st.markdown(f"**Metrics analyzed:** {', '.join(str(m) for m in metrics_list)}")
                            else:
                                st.markdown("**Metrics analyzed:** N/A")
                    except json.JSONDecodeError:
                        st.markdown("**Metrics analyzed:** Invalid data")
                    except Exception as e:
                        st.markdown(f"**Metrics analyzed:** Error loading ({str(e)})")
                    
                    st.markdown("---")
                    
                    # Display comparison table
                    st.markdown("**📊 Comparison Data:**")
                    try:
                        # Validate comparison data exists
                        if pd.isna(row['comparison_data']) or not row['comparison_data']:
                            st.warning("⚠️ Comparison data is missing for this comment")
                        else:
                            # Deserialize comparison data
                            try:
                                comparison_df = pd.read_json(io.StringIO(row['comparison_data']), orient='split')
                            except ValueError as e:
                                st.error(f"⚠️ Comparison data is corrupted: Invalid JSON format")
                                continue
                            except Exception as e:
                                st.error(f"⚠️ Failed to parse comparison data: {str(e)}")
                                continue
                            
                            # Validate comparison DataFrame is not empty
                            if comparison_df.empty:
                                st.warning("⚠️ Comparison data is empty")
                                continue
                            
                            # Format the comparison table for display using the same logic as Period Comparison tab
                            # This function matches the format_value function from Period Comparison tab
                            def format_value_for_display(val, metric):
                                if pd.isna(val):
                                    return "N/A"
                                try:
                                    # Currency columns (Amount, Spent, sales)
                                    if 'Amount' in metric or 'Spent' in metric or 'sales' in metric or 'Sales' in metric:
                                        return f"${val:,.2f}"
                                    # Percentage/rate columns
                                    elif 'rate' in metric or 'Rate' in metric or 'percentage' in metric or 'Percentage' in metric:
                                        return f"{val:.2%}"
                                    # Count columns - whole numbers (Impressions, Link Clicks, Sessions, Quantity, etc.)
                                    elif any(x in metric for x in ['Impressions', 'Link Clicks', 'Sessions', 'Quantity', 
                                                                    'ordered', 'customers', 'Customers', 'items', 'sold',
                                                                    'checkout', 'cart', 'additions']):
                                        return f"{val:,.0f}"
                                    # Default: 2 decimal places
                                    else:
                                        return f"{val:,.2f}"
                                except (ValueError, TypeError):
                                    return "N/A"
                            
                            display_data = []
                            try:
                                for _, comp_row in comparison_df.iterrows():
                                    metric_name = comp_row.get('Metric', 'Unknown')
                                    
                                    change_pct = comp_row.get('Change %', 0)
                                    try:
                                        change_pct_str = f"{change_pct:+.2f}%" if change_pct != float('inf') and not pd.isna(change_pct) else "N/A"
                                    except (ValueError, TypeError):
                                        change_pct_str = "N/A"
                                    
                                    display_row = {
                                        'Metric': metric_name,
                                        'Period 1': format_value_for_display(comp_row.get('Period 1', 0), metric_name),
                                        'Period 2': format_value_for_display(comp_row.get('Period 2', 0), metric_name),
                                        'Change': format_value_for_display(comp_row.get('Change', 0), metric_name),
                                        'Change %': change_pct_str
                                    }
                                    display_data.append(display_row)
                                
                                display_comparison_df = pd.DataFrame(display_data)
                                st.dataframe(display_comparison_df, use_container_width=True, hide_index=True)
                            except Exception as e:
                                st.error(f"⚠️ Error formatting comparison data: {str(e)}")
                    except Exception as e:
                        st.error(f"⚠️ Unexpected error displaying comparison data: {str(e)}")
                    
                    st.markdown("---")
                    
                    # Comment text display or edit mode
                    st.markdown("**💬 Comment:**")
                    
                    # Check if this comment is being edited
                    if st.session_state.editing_comment_id == comment_id:
                        # Edit mode
                        edit_text = st.text_area(
                            "Edit comment:",
                            value=st.session_state.edit_text,
                            height=150,
                            key=f"edit_area_{comment_id}"
                        )
                        
                        # Save and Cancel buttons
                        edit_col1, edit_col2, edit_col3 = st.columns([1, 1, 2])
                        with edit_col1:
                            if st.button("💾 Save", key=f"save_edit_{comment_id}", use_container_width=True):
                                if edit_text.strip():
                                    success, message = comment_manager.update_comment(comment_id, edit_text.strip())
                                    if success:
                                        st.session_state.comment_success_message = f"✅ {message}"
                                        # Clean up edit state
                                        st.session_state.editing_comment_id = None
                                        st.session_state.edit_text = ""
                                        st.rerun()
                                    else:
                                        st.session_state.comment_error_message = f"❌ {message}"
                                        st.rerun()
                                else:
                                    st.warning("⚠️ Comment cannot be empty")
                        
                        with edit_col2:
                            if st.button("❌ Cancel", key=f"cancel_edit_{comment_id}", use_container_width=True):
                                # Clean up edit state
                                st.session_state.editing_comment_id = None
                                st.session_state.edit_text = ""
                                st.rerun()
                    else:
                        # Display mode
                        st.markdown(f"_{row['comment_text']}_")
                    
                    st.markdown("---")
                    st.markdown("")  # Add spacing between comments
  
        # Tab 5 - Pivot Table
        with tab5:
            st.subheader("🔄 Pivot Table Analysis")
            st.markdown("Create custom pivot tables to analyze your data from different perspectives")
            
            # Two-column layout: Filters on left, Date Range on right
            filter_col, date_col = st.columns([1, 1])
            
            with filter_col:
                st.markdown("### 🔍 Filters")
                
                # Get all columns for filtering
                all_cols = df.columns.tolist()
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Add filter option
                enable_filter = st.checkbox("Enable column filters", value=False, key="piv_enable_filter")
                
                if enable_filter:
                    filter_columns = st.multiselect(
                        "Select columns to filter",
                        options=cat_cols,
                        default=[],
                        key="piv_filter_cols",
                        help="Choose columns to apply filters before creating pivot table"
                    )
                    
                    # Store filter conditions
                    filter_conditions = {}
                    
                    for col in filter_columns:
                        unique_vals = df[col].dropna().unique().tolist()
                        selected_vals = st.multiselect(
                            f"Filter by {col}",
                            options=unique_vals,
                            default=unique_vals,
                            key=f"piv_filter_{col}"
                        )
                        filter_conditions[col] = selected_vals
                    
                    # Apply filters
                    df_filtered = df.copy()
                    for col, vals in filter_conditions.items():
                        if vals:  # Only filter if values selected
                            df_filtered = df_filtered[df_filtered[col].isin(vals)]
                    
                    if len(filter_conditions) > 0:
                        st.caption(f"🔍 Filtered to **{len(df_filtered):,}** rows")
                else:
                    df_filtered = df.copy()
            
            with date_col:
                st.markdown("### 📅 Date Range")
                
                # Date inputs
                dr_col1, dr_col2 = st.columns(2)
                
                with dr_col1:
                    piv_start = st.date_input(
                        "Start Date",
                        value=st.session_state.start_date,
                        min_value=min_date_raw.date(),
                        max_value=max_date_raw.date(),
                        key="piv_start"
                    )
                
                with dr_col2:
                    piv_end = st.date_input(
                        "End Date",
                        value=st.session_state.end_date,
                        min_value=min_date_raw.date(),
                        max_value=max_date_raw.date(),
                        key="piv_end"
                    )
                
                # Quick date buttons
                btn_col1, btn_col2, btn_col3 = st.columns(3)
                
                with btn_col1:
                    if st.button("30d", use_container_width=True, key="piv30"):
                        st.session_state.start_date = max_date_raw.date() - timedelta(days=30)
                        st.session_state.end_date = max_date_raw.date()
                        st.rerun()
                
                with btn_col2:
                    if st.button("90d", use_container_width=True, key="piv90"):
                        st.session_state.start_date = max_date_raw.date() - timedelta(days=90)
                        st.session_state.end_date = max_date_raw.date()
                        st.rerun()
                
                with btn_col3:
                    if st.button("All", use_container_width=True, key="pivall"):
                        st.session_state.start_date = min_date_raw.date()
                        st.session_state.end_date = max_date_raw.date()
                        st.rerun()
                
                # Filter by date
                if piv_start > piv_end:
                    st.error("⚠️ Start date must be before end date")
                    df_piv = df_filtered
                else:
                    df_piv = df_filtered[(df_filtered['Day'].dt.date >= piv_start) & 
                                        (df_filtered['Day'].dt.date <= piv_end)].copy()
                    
                    if len(df_piv) == 0:
                        st.warning("⚠️ No data in selected range")
                    else:
                        st.caption(f"📊 **{len(df_piv):,} rows** from {piv_start} to {piv_end}")
            
            st.divider()
            
            # Get columns for pivot configuration
            all_c = df_piv.columns.tolist()
            num_c = df_piv.select_dtypes(include=[np.number]).columns.tolist()
            
            # Validate session state defaults against current dataframe
            saved_rows = st.session_state.get('piv_rows', [])
            saved_cols = st.session_state.get('piv_cols', [])
            saved_vals = st.session_state.get('piv_vals', [])
            
            # Only keep defaults that exist in current dataframe
            valid_rows = [r for r in saved_rows if r in all_c]
            valid_cols = [c for c in saved_cols if c in all_c]
            valid_vals = [v for v in saved_vals if v in num_c]
            
            # Configuration and Results - Side by Side
            cfg_col, result_col = st.columns([1, 2.5])
            
            with cfg_col:
                st.markdown("### ⚙️ Configuration")
                
                # Row Fields
                rows = st.multiselect(
                    "🎯 Row Fields *",
                    options=all_c,
                    default=valid_rows,
                    key="pr",
                    help="Required: Fields displayed as rows"
                )
                
                # Column Fields
                cols = st.multiselect(
                    "📊 Column Fields",
                    options=all_c,
                    default=valid_cols,
                    key="pc",
                    help="Optional: Leave empty for simple grouping"
                )
                
                # Value Fields
                vals = st.multiselect(
                    "💰 Value Fields *",
                    options=num_c,
                    default=valid_vals,
                    key="pv",
                    help="Required: Metrics to aggregate"
                )
                
                # Aggregation Method
                agg = st.selectbox(
                    "⚙️ Aggregation Method",
                    options=["Sum", "Mean", "Count", "Min", "Max", "Median"],
                    index=0,
                    key="pagg"
                )
                
                # Display Options
                st.markdown("**📊 Display Options**")
                show_pct = st.checkbox("Show as percentages", value=False, key="ppct")
                
                # Percentage base selection (only shown when percentages enabled)
                if show_pct:
                    pct_base = st.radio(
                        "Percentage base",
                        options=['Column Total', 'Grand Total', 'Row Total'],
                        index=1,  # Default to Grand Total
                        key="piv_pct_base",
                        help="Column Total: Each column sums to 100% | Grand Total: All values sum to 100% | Row Total: Each row sums to 100%"
                    )
                
                # Validation
                errors = []
                if not rows:
                    errors.append("Select row field(s)")
                if not vals:
                    errors.append("Select value field(s)")
                
                if errors:
                    for err in errors:
                        st.warning(f"⚠️ {err}")
                
                # Generate Button
                generate_disabled = len(errors) > 0
                
                if st.button(
                    "🔄 Generate Pivot Table",
                    disabled=generate_disabled,
                    type="primary",
                    use_container_width=True,
                    key="pgen"
                ):
                    st.session_state.piv_rows = rows
                    st.session_state.piv_cols = cols
                    st.session_state.piv_vals = vals
                    st.session_state.piv_gen = True
                    # Clear cached pivot to force regeneration
                    if 'piv_base_df' in st.session_state:
                        del st.session_state.piv_base_df
                    st.rerun()
            
            # Results Column
            with result_col:
                st.markdown("### 📋 Pivot Table Results")
                
                if st.session_state.get('piv_gen', False):
                    try:
                        # Generate pivot only once and cache it
                        if 'piv_base_df' not in st.session_state:
                            # Map aggregation methods
                            agg_map = {
                                'Sum': 'sum',
                                'Mean': 'mean',
                                'Count': 'count',
                                'Min': 'min',
                                'Max': 'max',
                                'Median': 'median'
                            }
                            agg_dict = {v: agg_map[agg] for v in vals}
                            
                            # Generate pivot table
                            if not cols:
                                # Simple groupby when no column fields
                                piv_df = df_piv.groupby(rows).agg(agg_dict).reset_index()
                                
                                # Add totals row
                                tot = {r: 'Total' if i == 0 else '' for i, r in enumerate(rows)}
                                for v in vals:
                                    tot[v] = df_piv[v].agg(agg_map[agg])
                                piv_df = pd.concat([piv_df, pd.DataFrame([tot])], ignore_index=True)
                            else:
                                # Full pivot table with columns
                                piv_df = pd.pivot_table(
                                    df_piv,
                                    values=vals,
                                    index=rows,
                                    columns=cols,
                                    aggfunc=agg_dict,
                                    margins=True,
                                    margins_name='Total',
                                    fill_value=0
                                )
                                
                                # Flatten MultiIndex columns if they exist
                                if isinstance(piv_df.columns, pd.MultiIndex):
                                    piv_df.columns = ['_'.join(str(col).strip() for col in cols if str(col) != '') 
                                                    for cols in piv_df.columns.values]
                                
                                # Reset index to make it manageable
                                piv_df = piv_df.reset_index()
                                
                                # Remove the grand "Total" column if there are multiple value fields
                                # (Can't meaningfully combine different metrics like Impressions + Amount Spent)
                                if len(vals) > 1 and 'Total' in piv_df.columns:
                                    piv_df = piv_df.drop(columns=['Total'])
                            
                            # Apply percentage conversion if enabled
                            if show_pct:
                                try:
                                    pct_base = st.session_state.get('piv_pct_base', 'Grand Total')
                                    
                                    # Identify row label columns
                                    row_label_cols = [col for col in piv_df.columns if col in rows]
                                    
                                    # Get all numeric columns
                                    numeric_cols_all = piv_df.select_dtypes(include=[np.number]).columns.tolist()
                                    
                                    # Separate data columns from Total columns
                                    # Total columns contain "_Total" or end with "Total"
                                    data_cols = []
                                    total_cols = []
                                    
                                    for col in numeric_cols_all:
                                        if col in rows:
                                            continue
                                        
                                        # Check if this is a Total column
                                        col_str = str(col)
                                        if '_Total' in col_str or col_str.endswith('Total'):
                                            total_cols.append(col)
                                        else:
                                            data_cols.append(col)
                                    
                                    # Separate data rows from Total row
                                    has_total_row = len(rows) > 0 and 'Total' in piv_df[rows[0]].values
                                    
                                    if has_total_row:
                                        data_df = piv_df[piv_df[rows[0]] != 'Total'].copy()
                                        total_row_df = piv_df[piv_df[rows[0]] == 'Total'].copy()
                                    else:
                                        data_df = piv_df.copy()
                                        total_row_df = None
                                    
                                    # Apply percentage calculations based on selected base
                                    if pct_base == 'Column Total':
                                        # Each column sums to 100%
                                        for col in data_cols:
                                            col_sum = data_df[col].sum()
                                            if col_sum > 0:
                                                data_df[col] = (data_df[col] / col_sum) * 100
                                        
                                        # Update total row to show 100% for each data column
                                        if total_row_df is not None:
                                            for col in data_cols:
                                                total_row_df[col] = 100.0
                                        
                                        # Recalculate Total columns as sums of their corresponding data columns
                                        for total_col in total_cols:
                                            # Find which value field this Total belongs to
                                            value_field_name = total_col.replace('_Total', '')
                                            
                                            # Find all data columns for this value field
                                            related_cols = [c for c in data_cols if c.startswith(value_field_name + '_')]
                                            
                                            if related_cols:
                                                # Sum the percentages across related columns
                                                data_df[total_col] = data_df[related_cols].sum(axis=1)
                                                
                                                if total_row_df is not None:
                                                    # Total row shows sum of percentages
                                                    total_row_df[total_col] = len(related_cols) * 100.0
                                    
                                    elif pct_base == 'Grand Total':
                                        # All values as % of grand total (ONLY data columns, not Total columns)
                                        grand_total = data_df[data_cols].sum().sum()
                                        
                                        if grand_total > 0:
                                            # Convert data columns to percentages
                                            for col in data_cols:
                                                data_df[col] = (data_df[col] / grand_total) * 100
                                        
                                        # Recalculate Total columns as sums of their corresponding data columns
                                        for total_col in total_cols:
                                            # Find which value field this Total belongs to
                                            value_field_name = total_col.replace('_Total', '')
                                            
                                            # Find all data columns for this value field
                                            related_cols = [c for c in data_cols if c.startswith(value_field_name + '_')]
                                            
                                            if related_cols:
                                                # Sum the percentages across related columns
                                                data_df[total_col] = data_df[related_cols].sum(axis=1)
                                        
                                        # Total row shows column sums
                                        if total_row_df is not None:
                                            for col in data_cols:
                                                total_row_df[col] = data_df[col].sum()
                                            
                                            # Recalculate Total columns in total row
                                            for total_col in total_cols:
                                                value_field_name = total_col.replace('_Total', '')
                                                related_cols = [c for c in data_cols if c.startswith(value_field_name + '_')]
                                                
                                                if related_cols:
                                                    total_row_df[total_col] = total_row_df[related_cols].sum()
                                    
                                    elif pct_base == 'Row Total':
                                        # Each row sums to 100% (across data columns only)
                                        for idx in data_df.index:
                                            row_sum = data_df.loc[idx, data_cols].sum()
                                            if row_sum > 0:
                                                for col in data_cols:
                                                    data_df.loc[idx, col] = (data_df.loc[idx, col] / row_sum) * 100
                                        
                                        # Recalculate Total columns as sums of their corresponding data columns
                                        for total_col in total_cols:
                                            value_field_name = total_col.replace('_Total', '')
                                            related_cols = [c for c in data_cols if c.startswith(value_field_name + '_')]
                                            
                                            if related_cols:
                                                data_df[total_col] = data_df[related_cols].sum(axis=1)
                                        
                                        # Total row shows average percentage for each column
                                        if total_row_df is not None:
                                            for col in data_cols:
                                                total_row_df[col] = data_df[col].mean()
                                            
                                            # Recalculate Total columns in total row
                                            for total_col in total_cols:
                                                value_field_name = total_col.replace('_Total', '')
                                                related_cols = [c for c in data_cols if c.startswith(value_field_name + '_')]
                                                
                                                if related_cols:
                                                    total_row_df[total_col] = total_row_df[related_cols].sum()
                                    
                                    # Recombine data and total rows
                                    if total_row_df is not None:
                                        piv_df = pd.concat([data_df, total_row_df], ignore_index=True)
                                    else:
                                        piv_df = data_df
                                
                                except Exception as e:
                                    st.warning(f"⚠️ Could not convert to percentages: {str(e)}")
                            
                            # Cache the base pivot
                            st.session_state.piv_base_df = piv_df.copy()
                        
                        # Use cached pivot
                        piv_df = st.session_state.piv_base_df.copy()
                        
                        # Get all columns for controls
                        piv_columns = piv_df.columns.tolist()
                        
                        # Table Controls at Top - More compact layout
                        st.markdown("---")
                        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.5, 1.5, 1])
                        
                        with ctrl_col1:
                            sort_col = st.selectbox(
                                "🔽 Sort by",
                                options=['None'] + piv_columns,
                                index=0,
                                key="piv_sort_col",
                                help="Sort table by selected column"
                            )
                        
                        with ctrl_col2:
                            if sort_col != 'None':
                                sort_order = st.selectbox(
                                    "Order",
                                    options=['↓ Descending', '↑ Ascending'],
                                    index=0,
                                    key="piv_sort_order"
                                )
                            else:
                                st.selectbox(
                                    "Order",
                                    options=['↓ Descending', '↑ Ascending'],
                                    index=0,
                                    disabled=True,
                                    key="piv_sort_order_disabled"
                                )
                        
                        with ctrl_col3:
                            hidden_cols = st.multiselect(
                                "🙈 Hide columns",
                                options=piv_columns,
                                default=[],
                                key="piv_hidden_cols",
                                help="Select columns to hide"
                            )
                        
                        with ctrl_col4:
                            hide_idx = st.checkbox(
                                "Hide index",
                                value=True,
                                key="piv_hide_idx"
                            )
                        
                        st.markdown("---")
                        
                        # Apply transformations to display copy only
                        display_df = piv_df.copy()
                        
                        # Apply sorting
                        if sort_col != 'None' and sort_col in display_df.columns:
                            sort_order = st.session_state.get('piv_sort_order', '↓ Descending')
                            ascending = (sort_order == '↑ Ascending')
                            
                            # Handle sorting with Total row
                            if len(rows) > 0 and 'Total' in display_df[rows[0]].values:
                                # Separate Total row
                                total_row_df = display_df[display_df[rows[0]] == 'Total'].copy()
                                data_rows_df = display_df[display_df[rows[0]] != 'Total'].copy()
                                
                                # Sort data rows
                                data_rows_df = data_rows_df.sort_values(by=sort_col, ascending=ascending)
                                
                                # Recombine with Total at the end
                                display_df = pd.concat([data_rows_df, total_row_df], ignore_index=True)
                            else:
                                display_df = display_df.sort_values(by=sort_col, ascending=ascending)
                        
                        # Apply column hiding
                        visible_cols = [c for c in display_df.columns if c not in hidden_cols]
                        display_df = display_df[visible_cols]
                        
                        # Get numeric columns for highlighting and formatting
                        numeric_cols_display = display_df.select_dtypes(include=[np.number]).columns.tolist()
                        
                        # Info and Download Row
                        info_row1, info_row2 = st.columns([2.5, 1])
                        
                        with info_row1:
                            num_rows = len(display_df)
                            num_cols = len(display_df.columns)
                            st.caption(f"📊 **{num_rows:,}** rows × **{num_cols:,}** columns | 🎨 Highlighting: 🟢 Highest · 🔴 Lowest (in each column)")
                        
                        with info_row2:
                            csv = display_df.to_csv(index=not hide_idx)
                            current_date = datetime.now().strftime('%Y%m%d')
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name=f"pivot_table_{current_date}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        # Apply styling only at display time
                        def style_dataframe(df, numeric_cols, rows_list, show_pct):
                            """Apply styling efficiently"""
                            styled = df.style
                            
                            # Highlighting
                            if len(numeric_cols) > 0:
                                # Get data row indices (exclude Total)
                                if len(rows_list) > 0 and 'Total' in df[rows_list[0]].values:
                                    data_row_indices = df[df[rows_list[0]] != 'Total'].index.tolist()
                                else:
                                    data_row_indices = df.index.tolist()
                                
                                if len(data_row_indices) > 0:
                                    subset_rows = pd.IndexSlice[data_row_indices, numeric_cols]
                                    
                                    # Apply highlighting
                                    styled = styled.highlight_max(
                                        subset=subset_rows,
                                        color='#90EE90',
                                        axis=0
                                    ).highlight_min(
                                        subset=subset_rows,
                                        color='#FFB6C6',
                                        axis=0
                                    )
                            
                            # Bold Total row
                            if len(rows_list) > 0 and 'Total' in df[rows_list[0]].values:
                                def bold_total_row(row):
                                    is_total = (row.name in df.index and df.loc[row.name, rows_list[0]] == 'Total')
                                    return ['font-weight: bold' if is_total else '' for _ in row]
                                styled = styled.apply(bold_total_row, axis=1)
                            
                            # Formatting
                            if show_pct:
                                styled = styled.format({col: "{:.2f}%" for col in numeric_cols}, na_rep='N/A')
                            else:
                                styled = styled.format({col: "{:,.0f}" for col in numeric_cols}, na_rep='N/A')
                            
                            return styled
                        
                        # Apply styling
                        styled_df = style_dataframe(display_df, numeric_cols_display, rows, show_pct)
                        
                        # Display table
                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            height=550,
                            hide_index=hide_idx
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating pivot table: {str(e)}")
                        st.exception(e)
                        st.info("💡 Try different field combinations or check your data")
                else:
                    # Placeholder with instructions
                    st.info("👈 Configure settings and click **Generate Pivot Table**")
                    
                    st.markdown("""
                    **Quick Start:**
                    - Select **filters** to narrow down your data (optional)
                    - Choose **date range** to focus on specific period
                    - Pick **row fields** for table rows (required)
                    - Add **column fields** for cross-tabulation (optional)
                    - Select **value fields** to aggregate (required)
                    - Choose **aggregation method** (Sum, Mean, Count, etc.)
                    - Click **Generate** to create your pivot table
                    
                    **Percentage Options:**
                    - **Column Total**: Each column independently sums to 100%
                    - **Grand Total**: All values together sum to 100%
                    - **Row Total**: Each row independently sums to 100%
                    
                    **After generating:**
                    - **Sort** by any column (Total row stays at bottom)
                    - **Hide** columns you don't need
                    - **Highlighting** is enabled by default (🟢 highest, 🔴 lowest per column)
                    - **Download** results as CSV
                    
                    **Note:** When using multiple value fields, the combined "Total" column is automatically removed since different metrics can't be meaningfully summed.
                    """)
    
    # Tab 6 - Correlation Analysis
    with tab6:
        st.subheader("🔗 Correlation Analysis")
        st.markdown("Analyze relationships between different metrics")
        
        # Data Filters at the top
        with st.expander("🔍 Advanced Data Filters", expanded=False):
            # Get categorical columns (non-numeric, non-date)
            categorical_cols = [col for col in df_filtered.columns 
                              if col != 'Day' and df_filtered[col].dtype == 'object']
            
            if categorical_cols:
                # Filter controls
                col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
                with col_ctrl1:
                    st.markdown("**📊 Filter by categorical columns**")
                with col_ctrl2:
                    if st.button("✅ Select All", key="select_all_corr", use_container_width=True):
                        for col in categorical_cols[:6]:
                            st.session_state[f"filter_corr_{col}"] = list(df_filtered[col].dropna().unique())
                        st.rerun()
                with col_ctrl3:
                    if st.button("🗑️ Clear All", key="clear_all_corr", use_container_width=True):
                        for col in categorical_cols[:6]:
                            if f"filter_corr_{col}" in st.session_state:
                                st.session_state[f"filter_corr_{col}"] = []
                        st.rerun()
                
                st.divider()
                
                # Filters in grid
                filter_cols = st.columns(min(3, len(categorical_cols)))
                active_filters_corr = {}
                
                for idx, col in enumerate(categorical_cols[:6]):
                    with filter_cols[idx % 3]:
                        unique_vals = df_filtered[col].dropna().unique()
                        
                        # Sort options
                        try:
                            unique_vals_sorted = sorted(unique_vals)
                        except:
                            unique_vals_sorted = list(unique_vals)
                        
                        if len(unique_vals_sorted) > 0 and len(unique_vals_sorted) <= 100:
                            # Show count
                            st.caption(f"**{col}** ({len(unique_vals_sorted)} options)")
                            
                            selected_vals = st.multiselect(
                                f"{col}",
                                options=unique_vals_sorted,
                                key=f"filter_corr_{col}",
                                label_visibility="collapsed"
                            )
                            if selected_vals:
                                active_filters_corr[col] = selected_vals
                
                # Apply filters
                if active_filters_corr:
                    for col, vals in active_filters_corr.items():
                        df_filtered = df_filtered[df_filtered[col].isin(vals)]
                    
                    # Show filter summary
                    st.success(f"✅ Active filters: {len(active_filters_corr)}")
                    filter_summary = " | ".join([f"**{col}**: {len(vals)} selected" for col, vals in active_filters_corr.items()])
                    st.caption(filter_summary)
                    st.metric("Filtered Rows", f"{len(df_filtered):,}", delta=f"{len(df_filtered) - len(df):,}")
            else:
                st.info("No categorical columns found for filtering")
        
        # Get numeric columns
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns for correlation analysis")
        else:
            # Configuration
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### ⚙️ Configuration")
                
                # Select metrics for correlation
                corr_metrics = st.multiselect(
                    "Select metrics to analyze",
                    options=numeric_cols,
                    default=selected_metrics[:min(10, len(selected_metrics))],
                    key="corr_metrics",
                    help="Select metrics to include in correlation analysis"
                )
                
                # Correlation method
                corr_method = st.selectbox(
                    "Correlation Method",
                    options=["Pearson", "Spearman", "Kendall"],
                    index=0,
                    help="Pearson: Linear relationships | Spearman: Monotonic relationships | Kendall: Rank correlation"
                )
                
                # Minimum correlation threshold
                min_corr = st.slider(
                    "Minimum correlation to highlight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Highlight correlations above this threshold"
                )
                
                # Color scheme
                color_scheme = st.selectbox(
                    "Color Scheme",
                    options=["RdBu_r", "coolwarm", "viridis", "plasma"],
                    index=0
                )
            
            with col2:
                if len(corr_metrics) < 2:
                    st.info("👈 Select at least 2 metrics to see correlation analysis")
                else:
                    # Calculate correlation matrix
                    corr_df = df_filtered[corr_metrics].corr(method=corr_method.lower())
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_df.values,
                        x=corr_df.columns,
                        y=corr_df.columns,
                        colorscale=color_scheme,
                        zmid=0,
                        text=corr_df.values,
                        texttemplate='%{text:.2f}',
                        textfont={"size": 10},
                        colorbar=dict(title="Correlation")
                    ))
                    
                    fig.update_layout(
                        title=f"{corr_method} Correlation Matrix",
                        xaxis_title="",
                        yaxis_title="",
                        height=600,
                        width=800
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Strong correlations table
            if len(corr_metrics) >= 2:
                st.divider()
                st.markdown("### 📊 Strong Correlations")
                
                # Find strong correlations
                strong_corr = []
                for i in range(len(corr_df.columns)):
                    for j in range(i+1, len(corr_df.columns)):
                        corr_value = corr_df.iloc[i, j]
                        if abs(corr_value) >= min_corr:
                            strong_corr.append({
                                'Metric 1': corr_df.columns[i],
                                'Metric 2': corr_df.columns[j],
                                'Correlation': corr_value,
                                'Strength': 'Strong Positive' if corr_value > 0 else 'Strong Negative'
                            })
                
                if strong_corr:
                    strong_corr_df = pd.DataFrame(strong_corr)
                    strong_corr_df = strong_corr_df.sort_values('Correlation', key=abs, ascending=False)
                    
                    st.dataframe(
                        strong_corr_df.style.format({'Correlation': '{:.3f}'}),
                        use_container_width=True
                    )
                    
                    st.caption(f"Found {len(strong_corr)} correlations with |r| ≥ {min_corr}")
                else:
                    st.info(f"No correlations found with |r| ≥ {min_corr}")
        
        # Comments Section for Correlation Analysis
        st.divider()
        st.markdown("### 💬 Analysis Comments")
        st.markdown("Add notes and insights about the correlation patterns you've discovered")
        
        # Comment input
        with st.expander("✍️ Add New Comment", expanded=False):
            comment_text_corr = st.text_area(
                "Your observations",
                placeholder="E.g., Strong positive correlation between Impressions and Link Clicks suggests effective ad creative...",
                height=100,
                key="corr_comment_text"
            )
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            with col_btn1:
                if st.button("💾 Save Comment", use_container_width=True, key="save_corr_comment"):
                    if comment_text_corr and comment_text_corr.strip():
                        # Save correlation comment with current analysis context
                        try:
                            import comment_manager
                            
                            # Create a summary DataFrame of current correlation analysis
                            corr_summary = pd.DataFrame({
                                'Analysis Type': ['Correlation Analysis'],
                                'Method': [corr_method],
                                'Metrics Analyzed': [', '.join(corr_metrics)],
                                'Date Range': [f"{st.session_state.start_date} to {st.session_state.end_date}"],
                                'Min Correlation Threshold': [min_corr]
                            })
                            
                            success, message = comment_manager.save_comment(
                                comment_text=comment_text_corr,
                                period1_start=st.session_state.start_date,
                                period1_end=st.session_state.end_date,
                                period2_start=st.session_state.start_date,  # Same period for correlation
                                period2_end=st.session_state.end_date,
                                comparison_data=corr_summary,
                                agg_method=corr_method,
                                selected_metrics=corr_metrics
                            )
                            
                            if success:
                                st.success("✅ Comment saved successfully!")
                                st.session_state.corr_comment_text = ""
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"❌ Error saving comment: {str(e)}")
                    else:
                        st.warning("⚠️ Please enter a comment before saving")
            
            with col_btn2:
                if st.button("🗑️ Clear", use_container_width=True, key="clear_corr_comment"):
                    st.session_state.corr_comment_text = ""
                    st.rerun()
        
        # Display saved comments
        try:
            import comment_manager
            comments_df = comment_manager.load_comments()
            
            if comments_df is not None and not comments_df.empty:
                st.markdown("#### 📋 Saved Comments")
                
                # Filter to show only correlation-related comments (you can add a tag system later)
                for idx, row in comments_df.iterrows():
                    with st.expander(f"💬 {row['timestamp'][:19]} - {row['agg_method']}", expanded=False):
                        st.markdown(f"**Comment:** {row['comment_text']}")
                        st.caption(f"📅 Period: {row['period1_start']} to {row['period1_end']}")
                        
                        # Show metrics if available
                        try:
                            metrics = json.loads(row['metrics'])
                            st.caption(f"📊 Metrics: {', '.join(metrics[:5])}" + (f" (+{len(metrics)-5} more)" if len(metrics) > 5 else ""))
                        except:
                            pass
                        
                        # Delete button
                        if st.button(f"🗑️ Delete", key=f"del_corr_{row['id']}"):
                            success, message = comment_manager.delete_comment(row['id'])
                            if success:
                                st.success("✅ Comment deleted")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
        except Exception as e:
            st.info("💡 Comment system available - add your first comment above!")
    
    # Tab 7 - Cardinality View
    with tab7:
        st.subheader("🎯 Cardinality Analysis")
        st.markdown("Analyze unique values and data distribution across columns")
        
        # Get all columns
        all_cols = df_filtered.columns.tolist()
        
        # Calculate cardinality for each column
        cardinality_data = []
        for col in all_cols:
            unique_count = df_filtered[col].nunique()
            total_count = len(df_filtered[col])
            null_count = df_filtered[col].isnull().sum()
            null_pct = (null_count / total_count) * 100 if total_count > 0 else 0
            cardinality_ratio = (unique_count / total_count) * 100 if total_count > 0 else 0
            
            # Determine column type
            if pd.api.types.is_numeric_dtype(df_filtered[col]):
                col_type = "Numeric"
            elif pd.api.types.is_datetime64_any_dtype(df_filtered[col]):
                col_type = "DateTime"
            else:
                col_type = "Categorical"
            
            # Classify cardinality
            if unique_count == 1:
                cardinality_class = "Constant"
            elif unique_count == total_count:
                cardinality_class = "Unique"
            elif cardinality_ratio < 5:
                cardinality_class = "Low"
            elif cardinality_ratio < 50:
                cardinality_class = "Medium"
            else:
                cardinality_class = "High"
            
            cardinality_data.append({
                'Column': col,
                'Type': col_type,
                'Unique Values': unique_count,
                'Total Rows': total_count,
                'Cardinality %': cardinality_ratio,
                'Null Count': null_count,
                'Null %': null_pct,
                'Classification': cardinality_class
            })
        
        cardinality_df = pd.DataFrame(cardinality_data)
        
        # Configuration
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🔍 Filters")
            
            # Filter by type
            type_filter = st.multiselect(
                "Column Type",
                options=cardinality_df['Type'].unique().tolist(),
                default=cardinality_df['Type'].unique().tolist(),
                key="card_type_filter"
            )
            
            # Filter by classification
            class_filter = st.multiselect(
                "Cardinality Classification",
                options=cardinality_df['Classification'].unique().tolist(),
                default=cardinality_df['Classification'].unique().tolist(),
                key="card_class_filter"
            )
            
            # Sort by
            sort_by = st.selectbox(
                "Sort By",
                options=['Column', 'Unique Values', 'Cardinality %', 'Null %'],
                index=2
            )
            
            sort_order = st.radio(
                "Sort Order",
                options=['Descending', 'Ascending'],
                index=0
            )
        
        with col2:
            # Apply filters
            filtered_card_df = cardinality_df[
                (cardinality_df['Type'].isin(type_filter)) &
                (cardinality_df['Classification'].isin(class_filter))
            ]
            
            # Sort
            filtered_card_df = filtered_card_df.sort_values(
                by=sort_by,
                ascending=(sort_order == 'Ascending')
            )
            
            # Display table
            st.dataframe(
                filtered_card_df.style.format({
                    'Cardinality %': '{:.2f}%',
                    'Null %': '{:.2f}%',
                    'Unique Values': '{:,.0f}',
                    'Total Rows': '{:,.0f}',
                    'Null Count': '{:,.0f}'
                }).background_gradient(subset=['Cardinality %'], cmap='YlOrRd'),
                use_container_width=True,
                height=500
            )
            
            st.caption(f"Showing {len(filtered_card_df)} of {len(cardinality_df)} columns")
        
        # Visualizations
        st.divider()
        st.markdown("### 📊 Cardinality Distribution")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Bar chart of unique values
            fig1 = px.bar(
                filtered_card_df.head(20),
                x='Column',
                y='Unique Values',
                color='Classification',
                title='Top 20 Columns by Unique Values',
                labels={'Unique Values': 'Count'},
                color_discrete_map={
                    'Constant': '#999999',
                    'Low': '#41C185',
                    'Medium': '#FFBD59',
                    'High': '#458EE2',
                    'Unique': '#FFCF87'
                }
            )
            fig1.update_layout(
                xaxis_tickangle=-45, 
                height=400,
                plot_bgcolor='#F5F5F5',
                paper_bgcolor='#FFFFFF'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with viz_col2:
            # Pie chart of classification distribution
            class_counts = filtered_card_df['Classification'].value_counts()
            fig2 = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title='Cardinality Classification Distribution',
                color=class_counts.index,
                color_discrete_map={
                    'Constant': '#999999',
                    'Low': '#41C185',
                    'Medium': '#FFBD59',
                    'High': '#458EE2',
                    'Unique': '#FFCF87'
                }
            )
            fig2.update_layout(
                height=400,
                paper_bgcolor='#FFFFFF'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed view for selected column
        st.divider()
        st.markdown("### 🔍 Detailed Column Analysis")
        
        selected_col = st.selectbox(
            "Select a column to analyze",
            options=filtered_card_df['Column'].tolist(),
            key="card_detail_col"
        )
        
        if selected_col:
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            
            col_data = df_filtered[selected_col]
            
            with detail_col1:
                st.metric("Unique Values", f"{col_data.nunique():,}")
                st.metric("Total Rows", f"{len(col_data):,}")
            
            with detail_col2:
                st.metric("Null Count", f"{col_data.isnull().sum():,}")
                st.metric("Null %", f"{(col_data.isnull().sum() / len(col_data) * 100):.2f}%")
            
            with detail_col3:
                st.metric("Cardinality %", f"{(col_data.nunique() / len(col_data) * 100):.2f}%")
                st.metric("Data Type", str(col_data.dtype))
            
            # Value distribution
            st.markdown("#### Value Distribution (Top 20)")
            value_counts = col_data.value_counts().head(20)
            
            if len(value_counts) > 0:
                fig3 = px.bar(
                    x=value_counts.index.astype(str),
                    y=value_counts.values,
                    labels={'x': selected_col, 'y': 'Count'},
                    title=f'Top 20 Values in {selected_col}',
                    color_discrete_sequence=['#FFBD59']
                )
                fig3.update_layout(
                    xaxis_tickangle=-45, 
                    height=400,
                    plot_bgcolor='#F5F5F5',
                    paper_bgcolor='#FFFFFF'
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                # Show table
                value_dist_df = pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / len(col_data) * 100)
                })
                
                st.dataframe(
                    value_dist_df.style.format({
                        'Count': '{:,.0f}',
                        'Percentage': '{:.2f}%'
                    }),
                    use_container_width=True
                )

    # Tab 8 - Clustering Analysis
    with tab8:
        st.subheader("🔮 Clustering Analysis")
        st.markdown("Discover patterns and group similar data points using clustering algorithms")
        
        # Data Filters at the top
        with st.expander("🔍 Advanced Data Filters", expanded=False):
            # Get categorical columns (non-numeric, non-date)
            categorical_cols = [col for col in df_filtered.columns 
                              if col != 'Day' and df_filtered[col].dtype == 'object']
            
            if categorical_cols:
                # Filter controls
                col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
                with col_ctrl1:
                    st.markdown("**📊 Filter by categorical columns**")
                with col_ctrl2:
                    if st.button("✅ Select All", key="select_all_cluster", use_container_width=True):
                        for col in categorical_cols[:6]:
                            st.session_state[f"filter_cluster_{col}"] = list(df_filtered[col].dropna().unique())
                        st.rerun()
                with col_ctrl3:
                    if st.button("🗑️ Clear All", key="clear_all_cluster", use_container_width=True):
                        for col in categorical_cols[:6]:
                            if f"filter_cluster_{col}" in st.session_state:
                                st.session_state[f"filter_cluster_{col}"] = []
                        st.rerun()
                
                st.divider()
                
                # Filters in grid
                filter_cols = st.columns(min(3, len(categorical_cols)))
                active_filters_cluster = {}
                
                for idx, col in enumerate(categorical_cols[:6]):
                    with filter_cols[idx % 3]:
                        unique_vals = df_filtered[col].dropna().unique()
                        
                        # Sort options
                        try:
                            unique_vals_sorted = sorted(unique_vals)
                        except:
                            unique_vals_sorted = list(unique_vals)
                        
                        if len(unique_vals_sorted) > 0 and len(unique_vals_sorted) <= 100:
                            # Show count
                            st.caption(f"**{col}** ({len(unique_vals_sorted)} options)")
                            
                            selected_vals = st.multiselect(
                                f"{col}",
                                options=unique_vals_sorted,
                                key=f"filter_cluster_{col}",
                                label_visibility="collapsed"
                            )
                            if selected_vals:
                                active_filters_cluster[col] = selected_vals
                
                # Apply filters
                if active_filters_cluster:
                    for col, vals in active_filters_cluster.items():
                        df_filtered = df_filtered[df_filtered[col].isin(vals)]
                    
                    # Show filter summary
                    st.success(f"✅ Active filters: {len(active_filters_cluster)}")
                    filter_summary = " | ".join([f"**{col}**: {len(vals)} selected" for col, vals in active_filters_cluster.items()])
                    st.caption(filter_summary)
                    st.metric("Filtered Rows", f"{len(df_filtered):,}", delta=f"{len(df_filtered) - len(df):,}")
            else:
                st.info("No categorical columns found for filtering")
        
        # Get numeric columns
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns for clustering analysis")
        else:
            # Two column layout
            config_col, viz_col = st.columns([1, 2])
            
            with config_col:
                st.markdown("### ⚙️ Configuration")
                
                # Select features for clustering
                st.markdown("**📊 Select Features**")
                cluster_features = st.multiselect(
                    "Choose features for clustering",
                    options=numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))],
                    key="cluster_features"
                )
                
                if len(cluster_features) >= 2:
                    st.divider()
                    
                    # Clustering algorithm
                    st.markdown("**🔧 Algorithm**")
                    cluster_algo = st.selectbox(
                        "Clustering Method",
                        ["K-Means", "DBSCAN", "Hierarchical"],
                        key="cluster_algo"
                    )
                    
                    # Initialize variables
                    n_clusters = None
                    eps = None
                    min_samples = None
                    use_auto_clusters = False
                    
                    # Algorithm-specific parameters
                    if cluster_algo == "K-Means":
                        use_auto_clusters = st.checkbox(
                            "🎯 Auto-detect optimal clusters (Elbow Method)",
                            value=True,
                            key="use_auto_clusters",
                            help="Automatically determine the best number of clusters using the elbow method"
                        )
                        
                        if not use_auto_clusters:
                            n_clusters = st.slider(
                                "Number of Clusters",
                                min_value=2,
                                max_value=10,
                                value=3,
                                key="n_clusters"
                            )
                        else:
                            n_clusters = None  # Will be determined by elbow method
                            
                    elif cluster_algo == "DBSCAN":
                        eps = st.slider(
                            "Epsilon (neighborhood size)",
                            min_value=0.1,
                            max_value=5.0,
                            value=0.5,
                            step=0.1,
                            key="eps"
                        )
                        min_samples = st.slider(
                            "Min Samples",
                            min_value=2,
                            max_value=10,
                            value=5,
                            key="min_samples"
                        )
                    else:  # Hierarchical
                        use_auto_clusters = st.checkbox(
                            "🎯 Auto-detect optimal clusters (Elbow Method)",
                            value=True,
                            key="use_auto_clusters_hier",
                            help="Automatically determine the best number of clusters using the elbow method"
                        )
                        
                        if not use_auto_clusters:
                            n_clusters = st.slider(
                                "Number of Clusters",
                                min_value=2,
                                max_value=10,
                                value=3,
                                key="n_clusters_hier"
                            )
                        else:
                            n_clusters = None  # Will be determined by elbow method
                    
                    st.divider()
                    
                    # Visualization options
                    st.markdown("**📈 Visualization**")
                    if len(cluster_features) >= 2:
                        viz_x = st.selectbox("X-axis", cluster_features, index=0, key="viz_x")
                        viz_y = st.selectbox("Y-axis", cluster_features, index=1, key="viz_y")
                    
                    # Run clustering button
                    run_clustering = st.button("🚀 Run Clustering", use_container_width=True, type="primary")
            
            with viz_col:
                if len(cluster_features) < 2:
                    st.info("👈 Select at least 2 features to perform clustering")
                elif 'run_clustering' in locals() and run_clustering:
                    try:
                        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
                        from sklearn.preprocessing import StandardScaler
                        
                        # Prepare data
                        cluster_data = df_filtered[cluster_features].dropna()
                        
                        if len(cluster_data) < 3:
                            st.error("❌ Not enough data points for clustering (minimum 3 required)")
                        else:
                            # Standardize features
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(cluster_data)
                            
                            # Helper function to find optimal clusters using elbow method
                            def find_optimal_clusters(data, max_k=10):
                                """Find optimal number of clusters using elbow method"""
                                max_k = min(max_k, len(data) - 1)  # Can't have more clusters than data points
                                inertias = []
                                K_range = range(2, max_k + 1)
                                
                                for k in K_range:
                                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                    kmeans.fit(data)
                                    inertias.append(kmeans.inertia_)
                                
                                # Calculate the elbow point using the "elbow" heuristic
                                # Find the point of maximum curvature
                                if len(inertias) >= 2:
                                    # Normalize the values
                                    x = np.array(range(len(inertias)))
                                    y = np.array(inertias)
                                    
                                    # Calculate distances from each point to the line connecting first and last points
                                    p1 = np.array([x[0], y[0]])
                                    p2 = np.array([x[-1], y[-1]])
                                    
                                    distances = []
                                    for i in range(len(x)):
                                        p = np.array([x[i], y[i]])
                                        d = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
                                        distances.append(d)
                                    
                                    # The elbow is at the point with maximum distance
                                    elbow_idx = np.argmax(distances)
                                    optimal_k = list(K_range)[elbow_idx]
                                else:
                                    optimal_k = 3  # Default fallback
                                
                                return optimal_k, inertias, K_range
                            
                            # Store elbow plot data if auto-detect is used
                            elbow_data = None
                            
                            # Perform clustering
                            if cluster_algo == "K-Means":
                                # Determine number of clusters
                                if n_clusters is None:  # Auto-detect using elbow method
                                    with st.spinner("🔍 Finding optimal number of clusters..."):
                                        optimal_k, inertias, K_range = find_optimal_clusters(scaled_data)
                                        n_clusters = optimal_k
                                        elbow_data = (inertias, K_range, n_clusters)
                                
                                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                labels = model.fit_predict(scaled_data)
                                st.success(f"✅ K-Means clustering completed with {n_clusters} clusters")
                            
                            elif cluster_algo == "DBSCAN":
                                model = DBSCAN(eps=eps, min_samples=min_samples)
                                labels = model.fit_predict(scaled_data)
                                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                                n_noise = list(labels).count(-1)
                                st.success(f"✅ DBSCAN found {n_clusters_found} clusters and {n_noise} noise points")
                            
                            else:  # Hierarchical
                                # Determine number of clusters
                                if n_clusters is None:  # Auto-detect using elbow method
                                    with st.spinner("🔍 Finding optimal number of clusters..."):
                                        optimal_k, inertias, K_range = find_optimal_clusters(scaled_data)
                                        n_clusters = optimal_k
                                        elbow_data = (inertias, K_range, n_clusters)
                                
                                model = AgglomerativeClustering(n_clusters=n_clusters)
                                labels = model.fit_predict(scaled_data)
                                st.success(f"✅ Hierarchical clustering completed with {n_clusters} clusters")
                            
                            # Add cluster labels to data
                            cluster_data['Cluster'] = labels
                            
                            # Main clustering visualization (always at top)
                            fig = px.scatter(
                                cluster_data,
                                x=viz_x,
                                y=viz_y,
                                color='Cluster',
                                title=f"{cluster_algo} Clustering Results",
                                labels={'Cluster': 'Cluster ID'},
                                color_continuous_scale=['#FFBD59', '#41C185', '#458EE2', '#FFCF87', '#FFE7C2'] if cluster_algo == "DBSCAN" else None,
                                color_discrete_sequence=['#FFBD59', '#41C185', '#458EE2', '#FFCF87', '#FFE7C2', '#FF6B6B', '#4ECDC4', '#95E1D3']
                            )
                            
                            fig.update_layout(
                                height=600,
                                plot_bgcolor='#F5F5F5',
                                paper_bgcolor='#FFFFFF'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Additional charts and stats in a scrollable container
                            with st.container(height=600):
                                # Show elbow plot if auto-detect was used
                                if elbow_data is not None:
                                    inertias, K_range, optimal_k = elbow_data
                                    st.info(f"🎯 Elbow method suggests **{optimal_k} clusters** as optimal")
                                    
                                    fig_elbow = px.line(
                                        x=list(K_range),
                                        y=inertias,
                                        labels={'x': 'Number of Clusters', 'y': 'Inertia (Within-cluster sum of squares)'},
                                        title='Elbow Method - Optimal Cluster Selection',
                                        markers=True
                                    )
                                    # Highlight the optimal point
                                    fig_elbow.add_scatter(
                                        x=[optimal_k],
                                        y=[inertias[optimal_k - 2]],
                                        mode='markers',
                                        marker=dict(size=15, color='red', symbol='star'),
                                        name='Optimal K',
                                        showlegend=True
                                    )
                                    fig_elbow.update_layout(
                                        height=300,
                                        plot_bgcolor='#F5F5F5',
                                        paper_bgcolor='#FFFFFF'
                                    )
                                    st.plotly_chart(fig_elbow, use_container_width=True)
                                    st.divider()
                                
                                # Cluster statistics
                                st.markdown("### 📊 Cluster Statistics")
                                cluster_stats = cluster_data.groupby('Cluster')[cluster_features].mean()
                                
                                # Display statistics
                                st.dataframe(
                                    cluster_stats.style.background_gradient(cmap='YlOrRd', axis=1),
                                    use_container_width=True
                                )
                                
                                st.divider()
                                
                                # Cluster sizes
                                st.markdown("### 📈 Cluster Sizes")
                                cluster_sizes = cluster_data['Cluster'].value_counts().sort_index()
                                
                                fig_sizes = px.bar(
                                    x=cluster_sizes.index,
                                    y=cluster_sizes.values,
                                    labels={'x': 'Cluster ID', 'y': 'Number of Points'},
                                    title='Distribution of Data Points Across Clusters',
                                    color_discrete_sequence=['#FFBD59']
                                )
                                fig_sizes.update_layout(
                                    height=300,
                                    plot_bgcolor='#F5F5F5',
                                    paper_bgcolor='#FFFFFF'
                                )
                                st.plotly_chart(fig_sizes, use_container_width=True)
                            
                    except ImportError:
                        st.error("❌ scikit-learn is required for clustering. Please install it: `pip install scikit-learn`")
                    except Exception as e:
                        st.error(f"❌ Error during clustering: {str(e)}")
                        st.exception(e)
                else:
                    st.info("👈 Configure clustering parameters and click 'Run Clustering' to see results")
