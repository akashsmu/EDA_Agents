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

# Page Configuration
st.set_page_config(
    page_title="EDA Agents",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Look
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.25rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    .card-text {
        font-size: 0.9rem;
        color: #718096;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "data_raw" not in st.session_state:
        st.session_state["data_raw"] = None
    if "graph" not in st.session_state:
        st.session_state["graph"] = None


init_session_state()

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/data-configuration.png", width=80)
    st.title("Settings")
    
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Required for agents to work.")
    
    if st.button("Reset Session", type="primary"):
        st.session_state["data_raw"] = None
        st.session_state["graph"] = None
        st.rerun()

    st.markdown("---")
    st.caption("v1.0.0 | EDA Agents")


# --- Landing Page (No Data) ---
if st.session_state["data_raw"] is None:
    st.markdown("<div class='main-header'>🔎 EDA Agents</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Your Autonomous Data Analysis Partner</div>", unsafe_allow_html=True)

    # Capability Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>📊 Visualize</div>
            <div class='card-text'>Generate Plotly charts automatically by asking questions like "Show me a scatter plot of X vs Y".</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>🧹 Wrangle</div>
            <div class='card-text'>Clean and transform your data. "Fill missing values", "Drop column Z", or "Filter rows".</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>📝 Report</div>
            <div class='card-text'>Create comprehensive HTML reports with Sweetviz or specific analysis like correlation funnels.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Main File Uploader
    uploaded_file = st.file_uploader(
        "📂 Upload your dataset to begin", 
        type=["csv"], 
        help="Currently supports CSV files. Excel support coming soon."
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["data_raw"] = df.to_dict(orient="records")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")

# --- Main App (Data Loaded) ---
else:
    if not openai_api_key and "OPENAI_API_KEY" not in os.environ:
        st.warning("⚠️ Please provide an OpenAI API Key in the sidebar to continue.")
        st.stop()
    
    # Initialize Graph
    if st.session_state["graph"] is None:
        llm = ChatOpenAI(model="gpt-4", api_key=openai_api_key or os.environ["OPENAI_API_KEY"])
        st.session_state["graph"] = create_eda_graph(llm)

    df_preview = pd.DataFrame(st.session_state["data_raw"])
    
    st.markdown(f"### 🗃️ Data Overview ({df_preview.shape[0]} rows, {df_preview.shape[1]} cols)")
    st.dataframe(df_preview.head())

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat & Analysis", "🛠️ Toolkit", "📊 Deep Reports"])

    with tab1:
        st.markdown("#### Ask your data")
        user_input = st.chat_input("Ex: 'Plot a histogram of Age' or 'Drop rows with missing values'")
        
        # History container (simple for now)
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Display history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "image" in msg:
                    st.plotly_chart(msg["image"])
        
        if user_input:
            # Display user message
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("🤖 Agent is thinking..."):
                initial_state = {
                    "messages": [{"role": "user", "content": user_input}],
                    "data_raw": st.session_state["data_raw"]
                }
                result = st.session_state["graph"].invoke(initial_state, config={"configurable": {"thread_id": "1"}})
                
                final_output = result.get("final_output", {})
                
                # Check for updates to data
                if "wrangled_data" in final_output:
                    st.session_state["data_raw"] = final_output["wrangled_data"]
                    st.toast("✅ Data updated successfully!", icon="💾")
                    # Force rerun to update preview table? optional
                
                # Parse response
                response_content = ""
                response_image = None
                
                if "plotly_json" in final_output:
                    response_image = final_output["plotly_json"]
                    response_content = "Here is the visualization you requested."
                elif "code" in final_output:
                     # This logic assumes the agent returns code or content. 
                     # For visualization agent it returns 'plotly_json'.
                     # For wrangling it currently returns 'wrangled_data'.
                     if "wrangled_data" in final_output:
                         response_content = "I've modified the data as requested."
                
                # If there is also text output from the agent (not currently implemented fully in agents, they just return dicts)
                # We interpret the result.
                
                if "error" in final_output and final_output["error"]:
                    response_content = f"❌ Error: {final_output['error']}"
                
                if not response_content:
                    response_content = "Task completed."

                st.session_state["messages"].append({"role": "assistant", "content": response_content, "image": response_image})
                with st.chat_message("assistant"):
                    st.write(response_content)
                    if response_image:
                        st.plotly_chart(response_image)
                    if "code" in final_output:
                         with st.expander("Peek at the Code"):
                             st.code(final_output["code"], language="python")


    with tab2:
        st.subheader("Quick Analysis Tools")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📝 Explain Dataset"):
                content = explain_data.invoke({"data_raw": st.session_state["data_raw"]})
                st.info(content)
        with c2:
            if st.button("🔍 Missing Values"):
                res, artifact = visualize_missing.invoke({"data_raw": st.session_state["data_raw"]})
                st.write(res)
                for name, plot in artifact.items():
                    st.image(base64.b64decode(plot), caption=name)
        with c3:
            cols = pd.DataFrame(st.session_state["data_raw"]).columns
            target = st.selectbox("Target Column", cols)

    with tab3:
        st.subheader("Full Reports")
        if st.button("Generate Sweetviz Report 🍭"):
            with st.spinner("Generating..."):
                res, artifact = generate_sweetviz_report.invoke({"data_raw": st.session_state["data_raw"], "include_html": True})
                st.success("Report Generated!")
                st.components.v1.html(artifact["report_html"], height=800, scrolling=True)
