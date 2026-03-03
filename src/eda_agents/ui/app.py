import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

import streamlit as st
import pandas as pd
import json
import base64
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

from eda_agents.agents.graph import create_eda_graph
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

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/data-configuration.png", width=80)
    st.title("EDA Agents")
    
    navigation = st.radio(
        "Navigation",
        options=["🏠 Home", "💬 AI Chat Analysis", "📊 Visualize Data", "🧹 Wrangle Data", "📋 Deep Reports"],
        index=0 if st.session_state["data_raw"] is None else 1
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
        st.session_state["graph"] = create_eda_graph(llm)

    # Display history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "image" in msg and msg["image"]:
                st.plotly_chart(msg["image"])
            if "plan" in msg and msg["plan"]:
                with st.expander("📝 Recommended Plan", expanded=True):
                    st.markdown(msg["plan"])

    user_input = st.chat_input("Ask about your data (e.g., 'Show Age distribution', 'Drop columns X')")
    
    # HITL Approval Button (shown only if graph is at a breakpoint)
    if st.session_state.get("pending_approval"):
        if st.button("✅ Approve & Proceed", use_container_width=True):
            logger.info("User approved the plan.")
            user_input = "Approve and proceed" # Simulate approval message
            st.session_state["pending_approval"] = False

    if user_input:
        logger.info(f"User interaction: {user_input}")
        st.session_state["messages"].append({"role": "user", "content": user_input})
        # with st.chat_message("user"): # Removed to avoid double render after rerun
        #     st.write(user_input)

        with st.spinner("🤖 Thinking..."):
            # Prepare state
            # If we are resuming, we need to provide the approval message
            initial_state = {
                "messages": [{"role": "user", "content": user_input}],
                "data_raw": st.session_state["data_raw"],
                "hitl_enabled": st.session_state.get("hitl_enabled", False)
            }
            
            logger.info("Invoking graph...")
            # Use the same thread_id for persistence
            config = {"configurable": {"thread_id": "1"}}
            
            # Check if we have a state to resume from
            snapshot = st.session_state["graph"].get_state(config)
            
            if snapshot.next:
                logger.info(f"Resuming graph from breakpoint: {snapshot.next}")
                # If resuming, we pass None as input to just continue with the new message in history
                result = st.session_state["graph"].invoke(None, config=config)
            else:
                result = st.session_state["graph"].invoke(initial_state, config=config)
            
            # Check if we hit an interrupt
            new_snapshot = st.session_state["graph"].get_state(config)
            
            if new_snapshot.next:
                logger.info("Graph interrupted for approval.")
                # The state should contain the plan
                plan = new_snapshot.values.get("router_decision", {}).get("final_output", {}).get("plan") 
                # Note: because we nested agents, the plan is inside final_output of the main graph node
                
                # Actually, the sub-agent state is what has the plan. 
                # Let's check where the plan is stored in the result.
                # result is what comes out of the node.
                
                plan = result.get("final_output", {}).get("plan")
                
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": "I've analyzed your request and prepared a plan. Please review it below:",
                    "plan": plan
                })
                st.session_state["pending_approval"] = True
                st.rerun()
            
            final_output = result.get("final_output", {})
            
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
        
        # Tabs for different analysis views
        tab1, tab2, tab3 = st.tabs(["📝 Narrative Summary", "📊 Table Statistics", "🔍 Missing Data Audit"])
        
        with tab1:
            st.markdown("#### 📝 Narrative Overview")
            with st.spinner("Generating narrative summary..."):
                summary = explain_data.invoke({"data_raw": st.session_state["data_raw"]})
                st.info(summary)
                logger.info("Auto-run: explain_data completed.")

        with tab2:
            st.markdown("#### 📊 Descriptive Statistics")
            from eda_agents.tools.eda import describe_dataset
            with st.spinner("Calculating statistics..."):
                res, artifact = describe_dataset.invoke({"data_raw": st.session_state["data_raw"]})
                if "describe_df" in artifact:
                    df_stats = pd.DataFrame(artifact["describe_df"])
                    st.dataframe(df_stats, use_container_width=True)
                else:
                    st.write(res)
                logger.info("Auto-run: describe_dataset completed.")

        with tab3:
            st.markdown("#### 🔍 Missing Value Analysis")
            with st.spinner("Analyzing missing data..."):
                res, artifact = visualize_missing.invoke({"data_raw": st.session_state["data_raw"]})
                st.write(res)
                
                # Show plots in a grid or sequence
                for name, plot_data in artifact.items():
                    st.image(base64.b64decode(plot_data), caption=name.replace('_', ' ').title())
                logger.info("Auto-run: visualize_missing completed.")

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
                st.session_state["messages"].append({"role": "user", "content": "Remove duplicate rows"})
                st.rerun()
        with col2:
            if st.button("Drop Missing Values", use_container_width=True):
                logger.info("UI Button: Drop Missing Values clicked.")
                st.session_state["messages"].append({"role": "user", "content": "Drop all rows with missing values"})
                st.rerun()
        with col3:
            if st.button("Fill Missing (Mean/Mode)", use_container_width=True):
                logger.info("UI Button: Fill Missing Values clicked.")
                st.session_state["messages"].append({"role": "user", "content": "Fill missing values with mean for numeric columns and mode for categorical columns"})
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
                      res, artifact = generate_sweetviz_report.invoke({"data_raw": st.session_state["data_raw"], "include_html": True})
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


