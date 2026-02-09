import streamlit as st
import pandas as pd
import os
import json
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

from eda_agents.agents.graph import create_eda_graph
from eda_agents.tools.eda import (
    explain_data, 
    visualize_missing, 
    generate_correlation_funnel,
    generate_sweetviz_report,
    generate_dtale_report
)

# Page Conf
st.set_page_config(page_title="EDA Agents", layout="wide")

st.title("🤖 EDA Agents")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if not openai_api_key and "OPENAI_API_KEY" not in os.environ:
    st.warning("Please provide an OpenAI API Key.")
    st.stop()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state["data_raw"] = df.to_dict(orient="records")
    st.success(f"Loaded data: {df.shape}")
else:
    st.info("Please upload a CSV file.")
    st.stop()

# Initialize Graph
if "graph" not in st.session_state:
    llm = ChatOpenAI(model="gpt-4", api_key=openai_api_key or os.environ["OPENAI_API_KEY"])
    st.session_state["graph"] = create_eda_graph(llm)

# UI Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat & Plot", "🛠️ EDA Tools", "📊 Reports"])

with tab1:
    user_input = st.text_input("Ask something about your data:")
    if st.button("Run"):
        with st.spinner("Agent is working..."):
            initial_state = {
                "messages": [{"role": "user", "content": user_input}],
                "data_raw": st.session_state["data_raw"]
            }
            result = st.session_state["graph"].invoke(initial_state, config={"configurable": {"thread_id": "1"}})
            
            final_output = result.get("final_output", {})
            st.write("### Result")
            
            if "plotly_json" in final_output:
                st.plotly_chart(final_output["plotly_json"])
            
            if "code" in final_output:
                st.expander("Show Code").code(final_output["code"])
            
            if "error" in final_output and final_output["error"]:
                st.error(final_output["error"])

            if "wrangled_data" in final_output:
                 st.session_state["data_raw"] = final_output["wrangled_data"]
                 st.success("Data Wrangled successfully! Updated session state.")
                 st.dataframe(pd.DataFrame(final_output["wrangled_data"]).head())


with tab2:
    st.subheader("Quick EDA Tools")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Explain Data"):
             content = explain_data.invoke({"data_raw": st.session_state["data_raw"]})
             st.markdown(content)

    with col2:
        if st.button("Missing Values"):
             content, artifact = visualize_missing.invoke({"data_raw": st.session_state["data_raw"]})
             st.write(content)
             st.image(base64.b64decode(artifact["matrix_plot"]), caption="Matrix Plot")
             import base64
    
    with col3:
        target_col = st.selectbox("Select Target for Correlation", options=pd.DataFrame(st.session_state["data_raw"]).columns)
        if st.button("Correlation Funnel"):
             content, artifact = generate_correlation_funnel.invoke({
                 "data_raw": st.session_state["data_raw"],
                 "target": target_col
             })
             st.write(content)
             if artifact.get("plotly_figure"):
                 st.plotly_chart(artifact["plotly_figure"])

with tab3:
    st.subheader("Comprehensive Reports")
    if st.button("Generate Sweetviz Report"):
        with st.spinner("Generating report..."):
            content, artifact = generate_sweetviz_report.invoke({"data_raw": st.session_state["data_raw"], "include_html": True})
            st.success(content)
            if artifact.get("report_html"):
                st.components.v1.html(artifact["report_html"], height=800, scrolling=True)

    if st.button("Launch D-Tale"):
        with st.spinner("Launching D-Tale..."):
            content, artifact = generate_dtale_report.invoke({"data_raw": st.session_state["data_raw"]})
            st.success(content)
            st.markdown(f"[Open D-Tale]({artifact['dtale_url']})")
