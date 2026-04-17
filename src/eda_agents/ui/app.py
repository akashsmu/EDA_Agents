import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

import streamlit as st
import pandas as pd
import json
import base64
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage

from eda_agents.multiagents.supervisor import SupervisorAgent
from eda_agents.tools.eda import (
    explain_data, 
    visualize_missing, 
    generate_sweetviz_report,
)
from eda_agents.utils.logger import logger

# Page Configuration
st.set_page_config(
    page_title="EDA Agents",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Look (Light/Dark Mode Neutral)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.25rem;
        text-align: center;
        margin-bottom: 2rem;
        opacity: 0.8;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .card-text {
        font-size: 0.95rem;
        opacity: 0.8;
    }
    /* Fix for font visibility in cards */
    .card-content {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "data_raw" not in st.session_state:
        st.session_state["data_raw"] = None
    if "graph" not in st.session_state:
        st.session_state["graph"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


init_session_state()

@st.cache_data(show_spinner=False)
def generate_data_explanation(data_summary_text: str, api_key: str):
    if not api_key and "OPENAI_API_KEY" not in os.environ:
        return "To view the AI Executive Summary, please provide your OpenAI API Key in the sidebar."
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        sys_msg = SystemMessage(content="You are an expert Data Analyst. Given this dataset summary, write a succinct, data-grounded, clear, and detailed executive explanation of the dataset's characteristics, potential issues (missing data, skewness, high zeros, imbalance), and interesting patterns. Use 2-3 short paragraphs formatting with markdown.")
        human_msg = HumanMessage(content=data_summary_text[:4000])
        res = llm.invoke([sys_msg, human_msg])
        return res.content
    except Exception as e:
        return f"Failed to generate explanation: {e}"

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/data-configuration.png", width=80)
    st.title("EDA Agents")
    
    navigation = st.radio(
        "Navigation",
        options=["🏠 Home", "💬 AI Chat Analysis", "📊 Visualize Data", "🧹 Wrangle Data", "📋 Deep Reports"],
        index=0 if st.session_state["data_raw"] is None else 2
    )
    logger.debug(f"User navigated to: {navigation}")
    
    st.markdown("---")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Required for agents to work.")
    
    st.markdown("---")
    st.session_state["hitl_enabled"] = st.checkbox(
        "🤝 Human-in-the-Loop", 
        value=st.session_state.get("hitl_enabled", False),
        help="When enabled, agents will pause and ask for approval after generating a plan."
    )

    if st.button("🔄 Reset Session", use_container_width=True):
        logger.info("User requested session reset.")
        st.session_state["data_raw"] = None
        st.session_state["graph"] = None
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    st.caption("v1.1.0 | Project Foundation")


def render_chat_interface(openai_api_key):
    # Initialize Graph
    if st.session_state["graph"] is None:
        if not openai_api_key and "OPENAI_API_KEY" not in os.environ:
            st.error("Please provide an OpenAI API Key in the sidebar.")
            st.stop()
        logger.info("Initializing LangGraph for Chat.")
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key or os.environ["OPENAI_API_KEY"])
        agent = SupervisorAgent(llm)
        st.session_state["graph"] = agent.graph # Use the graph property for persistence

    # Display history
    for msg_idx, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "image" in msg and msg["image"]:
                st.plotly_chart(msg["image"], key=f"chat_img_{msg_idx}")
            if "plan" in msg and msg["plan"]:
                with st.expander("📝 Recommended Plan", expanded=True):
                    st.markdown(msg["plan"])

    user_input = st.chat_input("Ask about your data (e.g., 'Show Age distribution', 'Drop columns X')")
    
    preset_action = st.session_state.pop("preset_action", None)
    
    # HITL Approval Button (shown only if graph is at a breakpoint)
    if st.session_state.get("pending_approval"):
        if st.button("✅ Approve & Proceed", use_container_width=True):
            logger.info("User approved the plan.")
            preset_action = "Approve and proceed" # Simulate approval message
            st.session_state["pending_approval"] = False

    active_input = user_input or preset_action

    if active_input:
        logger.info(f"User interaction: {active_input}")
        st.session_state["messages"].append({"role": "user", "content": active_input})
        # with st.chat_message("user"): # Removed to avoid double render after rerun
        #     st.write(active_input)

        with st.status("🤖 Thinking Process", expanded=True) as status_container:
            # Prepare state
            # If we are resuming, we need to provide the approval message
            initial_state = {
                "messages": [HumanMessage(content=active_input)],
                "data_raw": st.session_state["data_raw"],
                "hitl_enabled": st.session_state.get("hitl_enabled", False)
            }
            
            logger.info("Invoking graph...")
            # Use the same thread_id for persistence
            config = {"configurable": {"thread_id": "1"}}
            
            # Check if we have a state to resume from
            snapshot = st.session_state["graph"].get_state(config)
            
            input_data = None if snapshot.next else initial_state
            
            if snapshot.next:
                logger.info(f"Resuming graph from breakpoint: {snapshot.next}")
                
            events = st.session_state["graph"].stream(input_data, config=config, stream_mode="updates")
            
            for event in events:
                for node_name, state_update in event.items():
                    if node_name == "supervisor":
                        next_worker = state_update.get("next_worker")
                        if next_worker and next_worker != "FINISH":
                            st.write(f"👔 **Supervisor** decided to use **{next_worker}**.")
                        elif next_worker == "FINISH":
                            st.write("👔 **Supervisor** marked task as complete.")
                    else:
                        worker_name = node_name.replace('_worker', '').title()
                        st.write(f"⚙️ **{worker_name} Agent** completed its execution.")
                        
            status_container.update(label="✅ Analysis Complete", state="complete", expanded=False)
            
            result = st.session_state["graph"].get_state(config).values
            
            # Check if we hit an interrupt
            new_snapshot = st.session_state["graph"].get_state(config)
            final_output = result.get("final_output", {})
            
            # Sub-graph (worker) interrupt check
            plan_exists = "plan" in final_output and final_output["plan"] is not None
            has_executed_results = "wrangled_data" in final_output or "plotly_json" in final_output
            is_worker_interrupted = plan_exists and not has_executed_results and st.session_state.get("hitl_enabled")
            
            if new_snapshot.next or is_worker_interrupted:
                logger.info("Graph interrupted for approval.")
                plan = final_output.get("plan")
                
                if not plan:
                    def _find_plan(obj):
                        if isinstance(obj, dict):
                            if "plan" in obj and obj["plan"]:
                                return obj["plan"]
                            for k, v in obj.items():
                                res = _find_plan(v)
                                if res:
                                    return res
                        elif isinstance(obj, list):
                            for item in obj:
                                res = _find_plan(item)
                                if res:
                                    return res
                        return None
                    
                    plan = _find_plan(result)
                    
                if not plan:
                    plan = f"*[Dev Debug]* Plan could not be found computationally.\nResult keys: `{list(result.keys())}`\nFinal Output keys: `{list(final_output.keys())}`"
                
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": "I've analyzed your request and prepared a plan. Please review it below:",
                    "plan": plan
                })
                st.session_state["pending_approval"] = True
                st.rerun()
            
            
            if "wrangled_data" in final_output:
                logger.info("Data wrangling update detected in graph output.")
                st.session_state["data_raw"] = final_output["wrangled_data"]
                st.toast("✅ Data updated!")
            
            response_content = "Task completed."
            response_image = None
            
            if "plotly_json" in final_output:
                logger.info("Visualization result received.")
                response_image = final_output["plotly_json"]
                response_content = "Here is the visualization:"
            elif "wrangled_data" in final_output:
                response_content = "I have updated the data as requested."
            
            if "error" in final_output and final_output["error"]:
                logger.error(f"Graph execution returned error: {final_output['error']}")
                response_content = f"❌ Error: {final_output['error']}"

            st.session_state["messages"].append({"role": "assistant", "content": response_content, "image": response_image})
            st.rerun() # Refresh to show new messages

# --- Home / Landing Page ---
if navigation == "🏠 Home":
    st.markdown("<div class='main-header'>🔎 EDA Agents</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Your Autonomous Data Analysis Partner</div>", unsafe_allow_html=True)

    if st.session_state["data_raw"] is None:
        col1, col2, col3 = st.columns(3)
        with col1:
             st.markdown("<div class='card'><div class='card-title'>📊 Visualize</div><div class='card-text'>Instant Plotly charts from natural language.</div></div>", unsafe_allow_html=True)
        with col2:
             st.markdown("<div class='card'><div class='card-title'>🧹 Wrangle</div><div class='card-text'>Clean, filter, and transform data automatically.</div></div>", unsafe_allow_html=True)
        with col3:
             st.markdown("<div class='card'><div class='card-title'>📋 Reports</div><div class='card-text'>Comprehensive Sweetviz and Missingno audits.</div></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        uploaded_file = st.file_uploader("📂 Upload a CSV to begin", type=["csv"])
        if uploaded_file:
            logger.info(f"File uploaded: {uploaded_file.name}")
            df = pd.read_csv(uploaded_file)
            st.session_state["data_raw"] = df.to_dict(orient="records")
            logger.info(f"Loaded {len(df)} rows into session state.")
            st.rerun()
    else:
        st.success("✅ Data Loaded! Use the sidebar to start analyzing.")
        df_preview = pd.DataFrame(st.session_state["data_raw"])
        st.dataframe(df_preview.head(10), use_container_width=True)
        st.info(f"Shape: {df_preview.shape[0]} rows, {df_preview.shape[1]} columns")

# --- AI Chat Analysis ---
elif navigation == "💬 AI Chat Analysis":
    if st.session_state["data_raw"] is None:
        st.warning("Please upload data first on the Home page.")
    else:
        st.markdown("### 💬 AI Data Assistant")
        render_chat_interface(openai_api_key)

# --- Visualize Data ---
elif navigation == "📊 Visualize Data":
    if st.session_state["data_raw"] is None:
        st.warning("Please upload data first.")
    else:
        st.markdown("### 📊 Automated Data Insights")
        
        # Get AI Data Explanation globally for this page
        with st.expander("🤖 AI Executive Data Summary", expanded=True):
            with st.spinner("Analyzing dataset patterns..."):
                raw_summary = explain_data.invoke({"data_raw": st.session_state["data_raw"]})
                ai_explanation = generate_data_explanation(raw_summary, openai_api_key)
                st.markdown(ai_explanation)
        
        # Tabs for different analysis views
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Narrative Summary", "📊 Table Statistics", "🔍 Missing Data Audit", "📈 Data Distributions"])
        
        with tab1:
            st.markdown("#### 📝 Narrative Overview")
            with st.spinner("Generating narrative summary..."):
                summary = explain_data.invoke({"data_raw": st.session_state["data_raw"]})
                st.markdown(summary)
                logger.info("Auto-run: explain_data completed.")

        with tab2:
            st.markdown("#### 📊 Descriptive Statistics")
            from eda_agents.tools.eda import describe_dataset
            with st.spinner("Calculating statistics..."):
                res, artifact = describe_dataset.func(data_raw=st.session_state["data_raw"])
                if "describe_df" in artifact:
                    df_stats = pd.DataFrame(artifact["describe_df"])
                    st.dataframe(df_stats.astype(str), use_container_width=True)
                else:
                    st.write(res)
                logger.info("Auto-run: describe_dataset completed.")

        with tab3:
            st.markdown("#### 🔍 Missing Value Analysis")
            st.write("This section provides a visual summary of the completeness of your dataset. Understanding missing data is crucial for determining how to clean or impute your dataset before model training or deep analysis.")
            
            df_missing = pd.DataFrame(st.session_state["data_raw"])
            total_cells = df_missing.size
            missing_cells = df_missing.isnull().sum().sum()
            missing_percent = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", f"{df_missing.shape[0]:,}")
            col2.metric("Total Columns", f"{df_missing.shape[1]:,}")
            col3.metric("Missing Cells (%)", f"{missing_percent:.2f}%")
            
            st.markdown("---")
            
            with st.spinner("Analyzing missing data patterns..."):
                res, artifact = visualize_missing.func(data_raw=st.session_state["data_raw"])
                
                tab_matrix, tab_bar, tab_heatmap = st.tabs(["🧩 Nullity Matrix", "📊 Missingness Bar", "🔥 Nullity Correlation Heatmap"])
                
                with tab_matrix:
                    st.markdown("**Nullity Matrix**: Visualizes the presence (solid) and absence (blank) of data across all columns. It helps reveal patterns of missingness across rows.")
                    if "matrix_plot" in artifact:
                        st.image(base64.b64decode(artifact["matrix_plot"]), use_container_width=True)
                
                with tab_bar:
                    st.markdown("**Missingness Bar**: A simple bar chart displaying the exact count of non-null values per column. Useful for quickly identifying entirely complete vs. heavily missing columns.")
                    if "bar_plot" in artifact:
                        st.image(base64.b64decode(artifact["bar_plot"]), use_container_width=True)
                        
                with tab_heatmap:
                    st.markdown("**Nullity Correlation Heatmap**: Shows how strongly the presence or absence of one variable affects the presence of another. Values close to 1 indicate if one is missing, the other is likely missing too.")
                    missing_cols_count = (df_missing.isnull().sum() > 0).sum()
                    if missing_cols_count < 2:
                        st.info("💡 **Note:** The Nullity Correlation Heatmap requires at least two columns with missing data to compute correlations. Since your dataset has fewer than two such columns, this plot will be empty.")
                    if "heatmap_plot" in artifact:
                        st.image(base64.b64decode(artifact["heatmap_plot"]), use_container_width=True)

                logger.info("Auto-run: visualize_missing completed.")

        with tab4:
            st.markdown("#### 📈 Feature Distributions")
            st.write("Explore the spread, central tendency, and frequencies of your dataset's features.")
            df_plot = pd.DataFrame(st.session_state["data_raw"])
            
            numeric_cols = df_plot.select_dtypes(include='number').columns.tolist()
            cat_cols = df_plot.select_dtypes(exclude='number').columns.tolist()
            all_cols = numeric_cols + cat_cols
            
            import plotly.express as px
            
            if all_cols:
                # Default to a few columns to show something immediately
                default_cols = numeric_cols[:2] + cat_cols[:1] if len(numeric_cols) >= 2 and len(cat_cols) >= 1 else all_cols[:3]
                selected_cols = st.multiselect("Select Features to Visualize:", all_cols, default=default_cols)
                
                for i, col in enumerate(selected_cols):
                    st.markdown("---")
                    
                    # Cycle through some colors for variety
                    colors = [px.colors.qualitative.Plotly, px.colors.qualitative.Prism, px.colors.qualitative.Vivid, px.colors.qualitative.Pastel]
                    col_color_scale = colors[i % len(colors)]
                    
                    if col in numeric_cols:
                        st.markdown(f"##### 🔹 Distribution Analysis of `{col}` (Numeric)")
                        
                        tab_hist, tab_box, tab_violin = st.tabs(["📊 Histogram", "📦 Box Plot", "🎻 Violin Plot"])
                        
                        with tab_hist:
                            fig_hist = px.histogram(
                                df_plot, 
                                x=col, 
                                title=f"Histogram of {col}", 
                                color_discrete_sequence=[col_color_scale[0]],
                                marginal="rug",
                                opacity=0.85
                            )
                            fig_hist.update_layout(
                                xaxis_title=f"{col} Values",
                                yaxis_title="Count / Frequency",
                                showlegend=False,
                                hovermode="x unified",
                                margin=dict(t=50, l=50, r=50, b=50),
                                title_font=dict(size=18),
                            )
                            st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{i}_{col}")
                            
                        with tab_box:
                            fig_box = px.box(
                                df_plot, 
                                x=col, 
                                title=f"Box Plot of {col}", 
                                color_discrete_sequence=[col_color_scale[1 % len(col_color_scale)]],
                                points="all" # Show all points for better outlier visibility
                            )
                            fig_box.update_layout(
                                xaxis_title=f"{col} Values",
                                margin=dict(t=50, l=50, r=50, b=50),
                                title_font=dict(size=18),
                            )
                            st.plotly_chart(fig_box, use_container_width=True, key=f"box_{i}_{col}")
                            
                        with tab_violin:
                            fig_violin = px.violin(
                                df_plot, 
                                x=col, 
                                title=f"Violin Plot of {col}", 
                                color_discrete_sequence=[col_color_scale[2 % len(col_color_scale)]],
                                box=True, 
                                points="all"
                            )
                            fig_violin.update_layout(
                                xaxis_title=f"{col} Values",
                                margin=dict(t=50, l=50, r=50, b=50),
                                title_font=dict(size=18),
                            )
                            st.plotly_chart(fig_violin, use_container_width=True, key=f"violin_{i}_{col}")
                        
                        series = df_plot[col].dropna()
                        if not series.empty:
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Mean", f"{series.mean():.2f}")
                            c2.metric("Median", f"{series.median():.2f}")
                            c3.metric("Std Dev", f"{series.std():.2f}")
                            c4.metric("Zeros (%)", f"{(series == 0).mean()*100:.2f}%")
                            
                        st.info(f"**Interpretation of `{col}` Graph**: The **Histogram** displays the frequency distribution across the numerical range. The **Box Plot** highlights the median, interquartile range (middle 50% of data), and potential outliers (points outside the whiskers). The **Violin Plot** combines a box plot with a kernel density plot giving a deeper understanding of the distribution's shape and modality.")
                        
                    else:
                        st.markdown(f"##### 🔸 Frequency Analysis of `{col}` (Categorical)")
                        
                        val_counts = df_plot[col].value_counts(dropna=False).reset_index()
                        val_counts.columns = [col, 'Count']
                        val_counts[col] = val_counts[col].fillna("Missing/NaN").astype(str)
                        
                        if len(val_counts) > 30:
                            st.warning(f"High cardinality detected ({len(val_counts)} unique values). Showing top 30.")
                            val_counts = val_counts.head(30)
                            
                        tab_bar, tab_pie = st.tabs(["📊 Bar Chart", "🥧 Pie Chart"])
                        
                        with tab_bar:
                            fig_bar = px.bar(
                                val_counts, 
                                x=col, 
                                y='Count', 
                                title=f"Frequency Count for {col}", 
                                color=col,
                                color_discrete_sequence=col_color_scale,
                                text_auto='.2s'
                            )
                            fig_bar.update_layout(
                                xaxis_title=f"Categories of {col}",
                                yaxis_title="Count / Number of Records",
                                xaxis={'categoryorder':'total descending'},
                                showlegend=False,
                                margin=dict(t=50, l=50, r=50, b=50),
                                title_font=dict(size=18),
                                xaxis_tickangle=-45,
                            )
                            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{i}_{col}")
                            
                        with tab_pie:
                            # If cardinality is huge, pie chart is messy. Just standard Plotly Pie
                            fig_pie = px.pie(
                                val_counts, 
                                names=col, 
                                values='Count', 
                                title=f"Proportion Distribution of {col}", 
                                color_discrete_sequence=col_color_scale,
                                hole=0.3 # Make it a donut chart
                            )
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                            fig_pie.update_layout(
                                margin=dict(t=50, l=50, r=50, b=50),
                                title_font=dict(size=18),
                            )
                            st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{i}_{col}")
                        
                        st.info(f"**Interpretation of `{col}` Graph**: The **Bar Chart** visualizes the total record count for each respective category, ranked by frequency. The **Pie (Donut) Chart** provides a proportional view, showing what percentage of the entirety each category represents. Notice which categories dominate, or if there is a severe class imbalance.")

# --- Wrangle Data ---
elif navigation == "🧹 Wrangle Data":
    if st.session_state["data_raw"] is None:
        st.warning("Please upload data first.")
    else:
        st.markdown("### 🧹 Data Wrangling")
        st.write("Perform common data cleaning operations or ask the AI to do it for you.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Remove Duplicate Rows", use_container_width=True):
                logger.info("UI Button: Remove Duplicates clicked.")
                st.session_state["preset_action"] = "Remove duplicate rows"
                st.rerun()
        with col2:
            if st.button("Drop Missing Values", use_container_width=True):
                logger.info("UI Button: Drop Missing Values clicked.")
                st.session_state["preset_action"] = "Drop all rows with missing values"
                st.rerun()
        with col3:
            if st.button("Fill Missing (Mean/Mode)", use_container_width=True):
                logger.info("UI Button: Fill Missing Values clicked.")
                st.session_state["preset_action"] = "Fill missing values with mean for numeric columns and mode for categorical columns"
                st.rerun()
        
        st.markdown("---")
        render_chat_interface(openai_api_key)

# --- Deep Reports ---
elif navigation == "📋 Deep Reports":
    if st.session_state["data_raw"] is None:
        st.warning("Please upload data first.")
    else:
        st.markdown("### 📋 Automated EDA Reports")
        
        with st.container(border=True):
            st.markdown("#### 🍭 Sweetviz Full Report")
            st.write("Create a high-density, interactive HTML report with target analysis.")
            if st.button("Generate Detailed Report", use_container_width=True):
                 logger.info("UI Button: Sweetviz Report clicked.")
                 with st.spinner("Analyzing..."):
                      res, artifact = generate_sweetviz_report.func(data_raw=st.session_state["data_raw"], include_html=True)
                      st.success("Report Ready!")
                      st.components.v1.html(artifact["report_html"], height=800, scrolling=True)

        st.markdown("---")
        with st.container(border=True):
            st.markdown("#### 🤖 Custom AI Report")
            st.write("Generate a custom report with specific wrangling, plots, and an AI-generated summary.")
            
            wrangle_options = st.multiselect(
                "Select Wrangling Methods:",
                ["Drop Duplicates", "Drop Missing Values", "Fill Missing Values (Forward Fill)", "Fill Missing Values (Backward Fill)"]
            )
            
            analysis_options = st.multiselect(
                "Select Analysis & Plots to Include:",
                ["Data Summary", "Descriptive Statistics", "Missing Values Plot", "Correlation Funnel"],
                default=["Data Summary"]
            )
            
            target_col = None
            if "Correlation Funnel" in analysis_options:
                df_cols = pd.DataFrame(st.session_state["data_raw"]).columns.tolist()
                target_col = st.selectbox("Select Target Column for Correlation Funnel:", df_cols)
                
            report_instructions = st.text_area("Additional Instructions (e.g. 'Focus on outliers in Age'):")
            
            if st.button("Generate Custom Report", use_container_width=True):
                if not openai_api_key and "OPENAI_API_KEY" not in os.environ:
                    st.error("Please provide an OpenAI API Key in the sidebar to generate AI summaries.")
                else:
                    logger.info("UI Button: Custom AI Report clicked.")
                    from eda_agents.tools.report import generate_custom_report
                    with st.spinner("Generating custom report (this may take a minute)..."):
                        llm = ChatOpenAI(model="gpt-4", api_key=openai_api_key or os.environ["OPENAI_API_KEY"])
                        result = generate_custom_report(
                            data_raw=st.session_state["data_raw"],
                            wrangling_methods=wrangle_options,
                            analysis_methods=analysis_options,
                            target_col=target_col,
                            instructions=report_instructions,
                            llm=llm
                        )
                    
                    st.success("Custom Report Generated!")
                    st.session_state["data_raw"] = result["cleaned_data"]
                    if wrangle_options:
                        st.info("Dataset has been updated in memory based on the selected wrangling methods.")
                    
                    st.markdown("### 📝 Executive Text Summary")
                    st.markdown(result["summary_markdown"])
                    
                    # Display Artifacts if any exist
                    arts = result.get("artifacts", {})
                    
                    if "descriptive_statistics" in arts:
                        st.markdown("#### Descriptive Statistics")
                        st.dataframe(pd.DataFrame(arts["descriptive_statistics"]["describe_df"]))
                        
                    if "missing_plots" in arts:
                        st.markdown("#### Missing Values Plots")
                        plots = arts["missing_plots"]
                        if "matrix_plot" in plots:
                            st.image(base64.b64decode(plots["matrix_plot"]), caption="Missingity Matrix")
                        if "bar_plot" in plots:
                            st.image(base64.b64decode(plots["bar_plot"]), caption="Missing Values Bar Plot")
                            
                    if "correlation_funnel" in arts:
                        st.markdown("#### Correlation Funnel")
                        funnel = arts["correlation_funnel"]
                        if funnel.get("plot_image"):
                            st.image(base64.b64decode(funnel["plot_image"]), caption="Correlation Funnel Plot")


