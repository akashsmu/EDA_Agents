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
    
    st.markdown("---")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Required for agents to work.")
    
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state["data_raw"] = None
        st.session_state["graph"] = None
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    st.caption("v1.1.0 | Project Foundation")


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
            df = pd.read_csv(uploaded_file)
            st.session_state["data_raw"] = df.to_dict(orient="records")
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
        
        # Initialize Graph
        if st.session_state["graph"] is None:
            if not openai_api_key and "OPENAI_API_KEY" not in os.environ:
                st.error("Please provide an OpenAI API Key in the sidebar.")
                st.stop()
            llm = ChatOpenAI(model="gpt-4", api_key=openai_api_key or os.environ["OPENAI_API_KEY"])
            st.session_state["graph"] = create_eda_graph(llm)

        # Display history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "image" in msg and msg["image"]:
                    st.plotly_chart(msg["image"])

        user_input = st.chat_input("Ask about your data (e.g., 'Show Age distribution', 'Drop columns X')")
        
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("🤖 Thinking..."):
                initial_state = {
                    "messages": [{"role": "user", "content": user_input}],
                    "data_raw": st.session_state["data_raw"]
                }
                result = st.session_state["graph"].invoke(initial_state, config={"configurable": {"thread_id": "1"}})
                
                final_output = result.get("final_output", {})
                
                if "wrangled_data" in final_output:
                    st.session_state["data_raw"] = final_output["wrangled_data"]
                    st.toast("✅ Data updated!")
                
                response_content = "Task completed."
                response_image = None
                
                if "plotly_json" in final_output:
                    response_image = final_output["plotly_json"]
                    response_content = "Here is the visualization:"
                elif "wrangled_data" in final_output:
                    response_content = "I have updated the data as requested."
                
                if "error" in final_output and final_output["error"]:
                    response_content = f"❌ Error: {final_output['error']}"

                st.session_state["messages"].append({"role": "assistant", "content": response_content, "image": response_image})
                with st.chat_message("assistant"):
                    st.write(response_content)
                    if response_image:
                        st.plotly_chart(response_image)
                    if "code" in final_output:
                         with st.expander("Show Python Code"):
                             st.code(final_output["code"])

# --- Visualize Data ---
elif navigation == "📊 Visualize Data":
    if st.session_state["data_raw"] is None:
        st.warning("Please upload data first.")
    else:
        st.markdown("### 📊 Visualization Toolkit")
        col1, col2 = st.columns(2)
        
        with col1:
             with st.container(border=True):
                 st.markdown("#### 📝 Data Summary")
                 st.write("Generate a narrative overview of your columns and values.")
                 if st.button("Run Explainer", use_container_width=True):
                      summary = explain_data.invoke({"data_raw": st.session_state["data_raw"]})
                      st.info(summary)

        with col2:
             with st.container(border=True):
                 st.markdown("#### 🔍 Missing Data Audit")
                 st.write("Analyze gaps and missing values using Missingno.")
                 if st.button("Run Audit", use_container_width=True):
                      res, artifact = visualize_missing.invoke({"data_raw": st.session_state["data_raw"]})
                      st.write(res)
                      for name, plot in artifact.items():
                          st.image(base64.b64decode(plot), caption=name.replace('_', ' ').title())

# --- Wrangle Data ---
elif navigation == "🧹 Wrangle Data":
    if st.session_state["data_raw"] is None:
        st.warning("Please upload data first.")
    else:
        st.markdown("### 🧹 Data Wrangling")
        st.info("Direct wrangling tools are being integrated. For now, use the **AI Chat Analysis** to clean data using natural language.")
        
        st.markdown("#### Suggested Actions:")
        col1, col2 = st.columns(2)
        with col1:
             if st.button("Remove Duplicate Rows", use_container_width=True):
                  # This is just a placeholder to show it can be done via Chat
                  st.session_state["messages"].append({"role": "user", "content": "Remove duplicate rows"})
                  st.info("Action sent to AI Chat. Switch to Chat tab to see results.")
        with col2:
             if st.button("Handle Missing Values (Fillna)", use_container_width=True):
                  st.session_state["messages"].append({"role": "user", "content": "Fill missing values with mean for numeric and mode for categorical"})
                  st.info("Action sent to AI Chat.")

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
                 with st.spinner("Analyzing..."):
                      res, artifact = generate_sweetviz_report.invoke({"data_raw": st.session_state["data_raw"], "include_html": True})
                      st.success("Report Ready!")
                      st.components.v1.html(artifact["report_html"], height=800, scrolling=True)

