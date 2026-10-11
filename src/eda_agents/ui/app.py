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
from eda_agents.utils.plotting import plotly_from_dict

# Page Configuration
st.set_page_config(
    page_title="EDA Agents",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphic Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF904B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        text-align: center;
        margin-bottom: 2rem;
        opacity: 0.8;
        font-weight: 300;
    }
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.3);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.15);
    }
    .card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: inherit;
    }
    .card-text {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        opacity: 0.8;
        line-height: 1.5;
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
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "ml_session_result" not in st.session_state:
        st.session_state["ml_session_result"] = None
    if "dl_query" not in st.session_state:
        st.session_state["dl_query"] = ""


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
    
    if "navigation_radio" not in st.session_state:
        st.session_state["navigation_radio"] = "🏠 Home" if st.session_state.get("data_raw") is None else "📊 Visualize Data"
        
    navigation = st.radio(
        "Navigation",
        options=["🏠 Home", "💬 AI Chat Analysis", "📊 Visualize Data", "🧹 Wrangle Data",
                 "⚙️ Feature Engineering", "🤖 ML Training", "📁 Data Loader", "📋 Deep Reports"],
        key="navigation_radio"
    )
    logger.debug(f"User navigated to: {navigation}")
    
    st.markdown("---")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Required for agents to work.")
    if openai_api_key != st.session_state.get("openai_api_key"):
        st.session_state["openai_api_key"] = openai_api_key
        st.session_state["graph"] = None
    
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
        st.session_state["history"] = []
        st.rerun()

    st.markdown("---")
    
    if st.session_state.get("history"):
        st.markdown("### ⏳ State Timeline")
        st.caption("Click any event to roll back data state.")
        for idx, event in enumerate(st.session_state["history"]):
            is_active = (idx == len(st.session_state["history"]) - 1)
            btn_type = "primary" if is_active else "secondary"
            if st.button(f"{event['action']}\\n\\n{event['description']}", key=f"hist_btn_{idx}", use_container_width=True, type=btn_type):
                if not is_active:
                    st.session_state["data_raw"] = event["data"]
                    st.session_state["history"] = st.session_state["history"][:idx+1]
                    st.toast(f"Rolled back to: {event['action']}")
                    st.rerun()
                    
    st.markdown("---")
    st.caption("v1.2.0 | Feature Engineering + AutoML + Data Loader")

def get_visual_pipeline_html(active_node, message=""):
    nodes = [
        {"id": "supervisor", "label": "👔 Supervisor"},
        {"id": "cleaning_worker", "label": "🧹 Cleaning"},
        {"id": "wrangling_worker", "label": "🔨 Wrangling"},
        {"id": "visualization_worker", "label": "📊 Visualization"},
        {"id": "eda_worker", "label": "🔬 EDA Tools"},
        {"id": "feature_engineering_worker", "label": "⚙️ Feature Eng."},
        {"id": "sklearn_ml_worker", "label": "🤖 ML Training"},
    ]
    
    html = '<div style="display: flex; flex-direction: column; gap: 15px; margin-bottom: 15px;">'
    html += '<div style="display: flex; justify-content: center; align-items: center; gap: 15px; flex-wrap: wrap;">'
    for node in nodes:
        is_active = (node["id"] == active_node)
        border_color = "#FF4B4B" if is_active else "rgba(128,128,128,0.3)"
        bg_color = "rgba(255,75,75,0.2)" if is_active else "transparent"
        box_shadow = "0 0 15px rgba(255,75,75,0.5)" if is_active else "none"
        scale = "scale(1.05)" if is_active else "scale(1)"
        opacity = "1" if is_active else "0.5"
        
        style = f"border: 2px solid {border_color}; background: {bg_color}; border-radius: 8px; padding: 10px 15px; font-family: 'Outfit', sans-serif; font-size: 0.9rem; box-shadow: {box_shadow}; transform: {scale}; opacity: {opacity}; transition: all 0.3s ease; text-align: center;"
        html += f'<div style="{style}">{node["label"]}</div>'
    html += '</div>'
    if message:
        html += f'<div style="text-align: center; font-family: \'Outfit\', sans-serif; font-size: 0.95rem; color: #FF4B4B; margin-top: 5px;"><em>{message}</em></div>'
    html += '</div>'
    return html

def render_health_check_dashboard():
    if not st.session_state.get("data_raw"):
        return
        
    df = pd.DataFrame(st.session_state["data_raw"])
    
    st.markdown("### 🏥 Dataset Health Check")
    
    issues_found = []
    
    # 1. Check Missing Data
    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    if not missing_cols.empty:
        max_missing_col = missing_cols.idxmax()
        max_missing_pct = (missing_cols.max() / len(df)) * 100
        issues_found.append({
            "type": "Missing Data",
            "icon": "🔍",
            "message": f"**{len(missing_cols)} columns** contain missing data (up to {max_missing_pct:.1f}% in `{max_missing_col}`).",
            "action": "Impute with Mean/Mode",
            "command": "Impute missing values with mean for numeric columns and mode for categorical columns"
        })
        
    # 2. Check Outliers (Numeric using IQR)
    numeric_cols = df.select_dtypes(include='number').columns
    outlier_issues = []
    for col in numeric_cols:
        if df[col].nunique() <= 10:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
            
        outliers_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers_count > 0:
            outlier_issues.append((col, outliers_count))
            
    if outlier_issues:
        # Sort by most outliers
        outlier_issues.sort(key=lambda x: x[1], reverse=True)
        top_col, top_count = outlier_issues[0]
        issues_found.append({
            "type": "Outliers Detected",
            "icon": "⚠️",
            "message": f"Column `{top_col}` has **{top_count} outliers** (outside 1.5x IQR).",
            "action": f"Fix {top_col} outliers",
            "command": f"Remove outliers from column {top_col} using the IQR method"
        })
        
    # 3. Check Class Imbalance (Categorical)
    cat_cols = df.select_dtypes(exclude='number').columns
    imbalance_issues = []
    for col in cat_cols:
        val_counts = df[col].value_counts(normalize=True)
        if not val_counts.empty and val_counts.iloc[0] > 0.8: # More than 80% is one class
            imbalance_issues.append((col, val_counts.iloc[0] * 100))
            
    if imbalance_issues:
        imbalance_issues.sort(key=lambda x: x[1], reverse=True)
        top_col, top_pct = imbalance_issues[0]
        issues_found.append({
            "type": "Class Imbalance",
            "icon": "📈",
            "message": f"Categorical column `{top_col}` is highly imbalanced ({top_pct:.1f}% single class).",
            "action": f"Analyze {top_col}",
            "command": f"Show a bar chart distribution of {top_col} and explain the imbalance"
        })
        
    if not issues_found:
        st.success("✅ **Dataset is Healthy!** No major missing values, extreme outliers, or severe class imbalances detected.")
        return
        
    def handle_health_action(cmd):
        st.session_state["preset_action"] = cmd
        st.session_state["navigation_radio"] = "💬 AI Chat Analysis"
        
    # Render Cards
    cols = st.columns(len(issues_found))
    for idx, (col, issue) in enumerate(zip(cols, issues_found)):
        with col:
            st.markdown(f"""
            <div class='card' style='padding: 1rem; border-left: 4px solid #FF4B4B;'>
                <div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>{issue['icon']} <b>{issue['type']}</b></div>
                <div style='font-size: 0.9rem; opacity: 0.8; margin-bottom: 1rem;'>{issue['message']}</div>
            </div>
            """, unsafe_allow_html=True)
            st.button(issue["action"], key=f"health_action_{idx}", use_container_width=True, on_click=handle_health_action, args=(issue["command"],))
                
    st.markdown("---")

EDA_SUGGESTIONS = [
    ("🧰 Tools", "What EDA tools do you have access to? Return a table."),
    ("🩺 Data quality", "Assess the data quality of this dataset and tell me what to fix first."),
    ("🎯 Outliers", "Detect outliers in the numeric columns and explain which ones matter."),
    ("🔗 Correlations", "Analyze correlations between numeric columns and flag multicollinearity."),
    ("📐 Distributions", "Profile the distributions of numeric columns and suggest transformations."),
    ("🏷️ Schema", "Run the smart schema tagger to classify column roles and identify ID/target columns."),
]

SEVERITY_ICONS = {"high": "🔴", "medium": "🟠", "low": "🟡", "info": "🔵"}


def _is_table(value):
    return (isinstance(value, dict) and value and all(isinstance(v, list) for v in value.values())
            and len({len(v) for v in value.values()}) == 1)


def _render_quality_report(art, key_prefix):
    c1, c2, c3 = st.columns(3)
    c1.metric("Quality Score", f"{art['score']}/100")
    c2.metric("Grade", art["grade"])
    c3.metric("Issues", len(art["issues"].get("check", [])))
    issues = pd.DataFrame(art["issues"])
    actionable = issues[issues["severity"].isin(["high", "medium", "low"])] if not issues.empty else issues
    for i, issue in actionable.head(8).iterrows():
        left, right = st.columns([4, 1])
        column = f" · `{issue['column']}`" if issue["column"] else ""
        left.markdown(f"{SEVERITY_ICONS[issue['severity']]} **{issue['check']}**{column} — {issue['detail']}")
        # One click hands the suggested fix to the supervisor, which routes it to the cleaning agent.
        right.button("🛠️ Fix", key=f"{key_prefix}_fix_{i}", help=issue["suggested_fix"],
                     on_click=lambda cmd=issue["suggested_fix"]: st.session_state.update(preset_action=cmd))
    info = issues[issues["severity"] == "info"] if not issues.empty else issues
    if not info.empty:
        st.caption("Notes: " + "; ".join(f"`{r.column}` {r.check.lower()}" for r in info.itertuples()))


def render_eda_artifacts(artifacts, tool_calls, key_prefix):
    """Renders the charts, images and tables returned by the EDA Tools Agent."""
    if tool_calls:
        st.caption("🧰 Tools used: " + " → ".join(f"`{t}`" for t in tool_calls))
    for tool_name, art in artifacts.items():
        if not isinstance(art, dict):
            continue
        with st.expander(f"🔬 {tool_name.replace('_', ' ').title()}", expanded=False):
            if "score" in art and "grade" in art:
                _render_quality_report(art, f"{key_prefix}_{tool_name}")
                continue
            if art.get("conclusion"):
                (st.success if art.get("significant") else st.info)(art["conclusion"])
            for key, value in art.items():
                widget_key = f"{key_prefix}_{tool_name}_{key}"
                if key == "plotly_figure" and isinstance(value, dict) and value.get("data"):
                    st.plotly_chart(plotly_from_dict(value), use_container_width=True, key=widget_key)
                elif (key.endswith("_plot") or key == "plot_image") and isinstance(value, str) and value:
                    st.image(base64.b64decode(value), caption=key.replace("_", " ").title(), use_container_width=True)
                elif key == "report_file" and value and os.path.exists(value) and not art.get("report_html"):
                    with open(value, encoding="utf-8") as f:
                        st.components.v1.html(f.read(), height=700, scrolling=True)
                elif key == "report_html" and value:
                    st.components.v1.html(value, height=700, scrolling=True)
                elif key == "dtale_url" and value:
                    st.markdown(f"[Open D-Tale explorer]({value})")
                elif _is_table(value):
                    st.markdown(f"**{key.replace('_', ' ').title()}**")
                    st.dataframe(pd.DataFrame(value).astype(str), use_container_width=True)


def _render_ml_results(ml: dict, key_prefix: str = "ml"):
    """Renders ML training results inline in the chat."""
    if not ml:
        return
    task = ml.get("task_type", "")
    algo = ml.get("algorithm", "")
    metrics = ml.get("metrics", {})
    feat_imp = ml.get("feature_importances", [])
    plotly_fig = ml.get("plotly_figure")

    with st.expander(f"🤖 Model Results — {algo} ({task})", expanded=True):
        m_cols = st.columns(len(metrics) if len(metrics) <= 4 else 4)
        for i, (k, v) in enumerate(metrics.items()):
            if isinstance(v, (int, float)):
                m_cols[i % 4].metric(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else v)

        if isinstance(metrics.get("classification_report"), str):
            with st.expander("Classification Report"):
                st.code(metrics["classification_report"])

        if plotly_fig and isinstance(plotly_fig, dict) and plotly_fig.get("data"):
            st.plotly_chart(plotly_from_dict(plotly_fig), use_container_width=True, key=f"{key_prefix}_fig")
        elif feat_imp:
            import plotly.express as px
            fi_df = pd.DataFrame(feat_imp).head(20)
            if "feature" in fi_df.columns and "importance" in fi_df.columns:
                fig = px.bar(fi_df.sort_values("importance"), x="importance", y="feature",
                             orientation="h", title="Feature Importances (Top 20)")
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_fi")

        train_size = ml.get("train_size")
        test_size = ml.get("test_size")
        if train_size is not None:
            st.caption(f"Train: {train_size:,} rows | Test: {test_size:,} rows | Target: `{ml.get('target_variable', '')}`")


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
                    st.markdown(msg["plan"], unsafe_allow_html=True)
            if "preview_data" in msg and msg["preview_data"] is not None:
                st.dataframe(msg["preview_data"])
            if msg.get("eda_artifacts") or msg.get("tool_calls"):
                render_eda_artifacts(msg.get("eda_artifacts") or {}, msg.get("tool_calls") or [], key_prefix=f"msg{msg_idx}")
            if msg.get("ml_results"):
                _render_ml_results(msg["ml_results"], key_prefix=f"ml_{msg_idx}")

    # Always rendered (not only on an empty chat) so the widget layout stays stable across reruns.
    suggestion_cols = st.columns(len(EDA_SUGGESTIONS))
    for col, (label, prompt) in zip(suggestion_cols, EDA_SUGGESTIONS):
        col.button(label, key=f"eda_suggest_{label}", use_container_width=True, help=prompt,
                   on_click=lambda p=prompt: st.session_state.update(preset_action=p))

    user_input = st.chat_input("Ask about your data (e.g., 'Is Fare related to Survived?', 'Find outliers', 'Drop columns X')")
    
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
                "hitl_enabled": st.session_state.get("hitl_enabled", False),
                "final_output": {}
            }
            
            logger.info("Invoking graph...")
            # Use the same thread_id for persistence
            config = {"configurable": {"thread_id": "1"}}
            
            # Check if we have a state to resume from
            snapshot = st.session_state["graph"].get_state(config)
            
            input_data = None if snapshot.next else initial_state
            
            if snapshot.next:
                logger.info(f"Resuming graph from breakpoint: {snapshot.next}")
                
            visual_placeholder = st.empty()
            visual_placeholder.markdown(get_visual_pipeline_html("supervisor", "Initializing..."), unsafe_allow_html=True)
                
            events = st.session_state["graph"].stream(input_data, config=config, stream_mode="updates")
            
            for event in events:
                for node_name, state_update in event.items():
                    msg = ""
                    if node_name == "supervisor":
                        next_worker = state_update.get("next_worker")
                        if next_worker and next_worker != "FINISH":
                            msg = f"Routing to {next_worker.replace('_worker', '').title()} Agent..."
                            st.write(f"👔 **Supervisor** decided to use **{next_worker}**.")
                        elif next_worker == "FINISH":
                            msg = "Task marked as complete."
                            st.write("👔 **Supervisor** marked task as complete.")
                    else:
                        worker_name = node_name.replace('_worker', '').title()
                        msg = f"{worker_name} Agent executing tasks..."
                        st.write(f"⚙️ **{worker_name} Agent** completed its execution.")
                        
                    visual_placeholder.markdown(get_visual_pipeline_html(node_name, msg), unsafe_allow_html=True)
                        
            status_container.update(label="✅ Analysis Complete", state="complete", expanded=False)
            
            result = st.session_state["graph"].get_state(config).values

            # Check if we hit an interrupt (outer supervisor graph only — inner agents no longer use interrupt_after)
            new_snapshot = st.session_state["graph"].get_state(config)
            final_output = result.get("final_output", {})

            # HITL: plan ready but code not yet executed (should_generate returned "wait")
            plan_exists = bool(final_output.get("plan"))
            has_executed_results = any(k in final_output for k in [
                "wrangled_data", "cleaned_data", "plotly_json",
                "engineered_data", "ml_results", "eda_artifacts", "ai_message",
            ])
            is_worker_interrupted = plan_exists and not has_executed_results and st.session_state.get("hitl_enabled")
            
            if new_snapshot.next or is_worker_interrupted:
                logger.info("Graph interrupted for HITL approval.")
                plan = final_output.get("plan") or ""

                if not plan:
                    plan = "*(Plan not available — please approve to proceed with the recommended steps.)*"
                
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": "I've analyzed your request and prepared a plan. Please review it below:",
                    "plan": plan
                })
                st.session_state["pending_approval"] = True
                st.rerun()
            
            
            # Detect which key holds updated data (supports all workers)
            data_update_key = (
                "wrangled_data" if "wrangled_data" in final_output else
                "cleaned_data" if "cleaned_data" in final_output else
                "engineered_data" if "engineered_data" in final_output else
                None
            )

            if data_update_key:
                logger.info(f"Data update ({data_update_key}) detected in graph output.")
                df_new = pd.DataFrame(final_output[data_update_key])
                for col in df_new.columns:
                    if df_new[col].dtype == 'object':
                        df_new[col] = df_new[col].apply(lambda x: str(x) if pd.notnull(x) else None)
                final_output[data_update_key] = df_new.to_dict(orient="records")

                prev_len = len(st.session_state["history"][-1]["data"]) if st.session_state.get("history") else len(st.session_state.get("data_raw") or [])
                new_len = len(final_output[data_update_key])
                diff = new_len - prev_len
                diff_str = f"({'+' if diff >= 0 else ''}{diff} rows)" if diff != 0 else "(values updated)"

                icons = {"wrangled_data": "🔨", "cleaned_data": "🧹", "engineered_data": "⚙️"}
                icon = icons.get(data_update_key, "📊")
                action_desc = f"{icon} {active_input[:30]}..." if active_input else f"{icon} Data update"

                st.session_state["data_raw"] = final_output[data_update_key]
                st.session_state["history"].append({
                    "action": action_desc,
                    "description": diff_str,
                    "data": final_output[data_update_key],
                })
                st.toast("✅ Data updated!")

            response_content = "Task completed."
            response_image = None
            response_preview = None
            response_ml_results = None

            if "eda_artifacts" in final_output:
                logger.info(f"EDA result received (tools: {final_output.get('tool_calls')}).")
                response_content = final_output.get("ai_message") or "EDA analysis complete."
            elif "ml_results" in final_output:
                logger.info("ML training result received.")
                ml = final_output["ml_results"]
                response_ml_results = ml
                task = ml.get("task_type", "")
                algo = ml.get("algorithm", "")
                metrics = ml.get("metrics", {})
                metric_str = " | ".join(f"**{k}**: {v}" for k, v in metrics.items() if isinstance(v, (int, float)))
                response_content = f"✅ Model trained! **{algo}** ({task})\n\n{metric_str}"
            elif "plotly_json" in final_output:
                logger.info("Visualization result received.")
                response_image = final_output["plotly_json"]
                response_content = "Here is the visualization:"
            elif data_update_key:
                icon = {"wrangled_data": "🔨", "cleaned_data": "🧹", "engineered_data": "⚙️"}.get(data_update_key, "📊")
                response_content = f"{icon} Data updated. Here is a sample:"
                response_preview = pd.DataFrame(final_output[data_update_key]).head(5)

            if "error" in final_output and final_output["error"]:
                logger.error(f"Graph execution returned error: {final_output['error']}")
                response_content = f"❌ Error: {final_output['error']}"

            st.session_state["messages"].append({
                "role": "assistant",
                "content": response_content,
                "image": response_image,
                "preview_data": response_preview,
                "eda_artifacts": final_output.get("eda_artifacts"),
                "tool_calls": final_output.get("tool_calls"),
                "ml_results": response_ml_results,
            })
            st.rerun()

def _export_pipeline_as_python() -> str:
    """Generate a reproducible Python script from the session history."""
    history = st.session_state.get("history", [])
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "",
        "# ============================================================",
        "# Auto-generated pipeline script from EDA Agents",
        "# Replay your data transformations in a standalone script.",
        "# ============================================================",
        "",
        "# Load your data",
        "df = pd.read_csv('your_data.csv')  # Replace with your file path",
        "",
    ]
    for i, event in enumerate(history[1:], start=1):
        action = event.get("action", f"Step {i}")
        desc = event.get("description", "")
        lines.append(f"# Step {i}: {action} {desc}")
        # Emit generic pandas equivalents based on action keywords
        action_lower = action.lower()
        if "duplicate" in action_lower:
            lines.append("df = df.drop_duplicates().reset_index(drop=True)")
        elif "drop missing" in action_lower or "drop_missing" in action_lower:
            lines.append("df = df.dropna().reset_index(drop=True)")
        elif "fill missing" in action_lower or "mean" in action_lower:
            lines.append("num_cols = df.select_dtypes(include='number').columns")
            lines.append("df[num_cols] = df[num_cols].fillna(df[num_cols].mean())")
            lines.append("cat_cols = df.select_dtypes(exclude='number').columns")
            lines.append("df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])")
        elif "feature eng" in action_lower or "engineering" in action_lower:
            lines.append("# Feature engineering was performed by the AI agent.")
            lines.append("# See the 'Feature Engineering' tab for the generated code.")
        else:
            lines.append(f"# (AI agent performed: {action})")
        lines.append("")

    lines += [
        "print('Pipeline complete.')",
        "print(f'Final shape: {df.shape}')",
        "df.head()",
    ]
    return "\n".join(lines)


# --- Home / Landing Page ---
if navigation == "🏠 Home":
    st.markdown("<div class='main-header'>🔎 EDA Agents</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Your Autonomous Data Analysis Partner</div>", unsafe_allow_html=True)

    if st.session_state["data_raw"] is None:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.markdown("<div class='card'><div class='card-title'>📊 Visualize</div><div class='card-text'>Instant Plotly charts from natural language.</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='card'><div class='card-title'>🧹 Wrangle</div><div class='card-text'>Clean, filter, and transform data automatically.</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='card'><div class='card-title'>🔬 EDA Tools</div><div class='card-text'>Quality scores, outliers, correlations and statistical tests.</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown("<div class='card'><div class='card-title'>⚙️ Feature Eng.</div><div class='card-text'>ML-ready encoding, scaling, and datetime features.</div></div>", unsafe_allow_html=True)
        with col5:
            st.markdown("<div class='card'><div class='card-title'>🤖 AutoML</div><div class='card-text'>Train sklearn models with feature importance analysis.</div></div>", unsafe_allow_html=True)
        with col6:
            st.markdown("<div class='card'><div class='card-title'>📋 Reports</div><div class='card-text'>Comprehensive Sweetviz and Missingno audits.</div></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        uploaded_file = st.file_uploader("📂 Upload a CSV to begin", type=["csv"])
        if uploaded_file:
            logger.info(f"File uploaded: {uploaded_file.name}")
            df = pd.read_csv(uploaded_file)
            st.session_state["data_raw"] = df.to_dict(orient="records")
            st.session_state["history"] = [{
                "action": f"📂 Uploaded {uploaded_file.name}",
                "description": f"{len(df)} rows",
                "data": st.session_state["data_raw"]
            }]
            logger.info(f"Loaded {len(df)} rows into session state.")
            st.rerun()

        if st.button("🎲 Or try a sample churn dataset", use_container_width=True,
                     help="Synthetic telecom churn data with built-in quality issues to explore."):
            from eda_agents.utils.sample_data import make_sample_churn_data
            df = make_sample_churn_data()
            st.session_state["data_raw"] = df.to_dict(orient="records")
            st.session_state["history"] = [{
                "action": "🎲 Sample churn data",
                "description": f"{len(df)} rows",
                "data": st.session_state["data_raw"]
            }]
            logger.info(f"Loaded sample dataset ({len(df)} rows).")
            st.rerun()
    else:
        st.success("✅ Data Loaded! Use the sidebar to start analyzing.")
        render_health_check_dashboard()
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
        render_health_check_dashboard()
        
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
                        action_desc = f"🧹 Deep Report Wrangling"
                        diff_str = f"Applied: {', '.join(wrangle_options)}"
                        st.session_state["history"].append({
                            "action": action_desc,
                            "description": diff_str,
                            "data": result["cleaned_data"]
                        })
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


# --- Feature Engineering ---
elif navigation == "⚙️ Feature Engineering":
    if st.session_state["data_raw"] is None:
        st.warning("Please upload data first on the Home page.")
    else:
        st.markdown("### ⚙️ Feature Engineering")
        st.write("Transform your dataset into ML-ready features: encode categoricals, create datetime features, drop ID columns, and scale numerics.")

        df_fe = pd.DataFrame(st.session_state["data_raw"])
        cols = df_fe.columns.tolist()

        with st.container(border=True):
            fe_col1, fe_col2 = st.columns(2)
            with fe_col1:
                target_var = st.selectbox("Target Variable (will NOT be transformed)", ["None"] + cols)
                target_var = None if target_var == "None" else target_var
            with fe_col2:
                fe_instructions = st.text_area(
                    "Custom Instructions (optional)",
                    placeholder="e.g. 'Do NOT scale the Age column' or 'Create interaction terms for Tenure × MonthlyCharges'",
                    height=80,
                )

        quick_cols = st.columns(3)
        fe_presets = [
            ("🔤 Encode Categoricals", "One-hot encode all categorical/object columns and drop the originals."),
            ("📅 Datetime Features", "Extract year, month, day_of_week, is_weekend from any datetime columns."),
            ("📏 Scale Numerics", "Apply StandardScaler to all numeric columns except the target."),
        ]
        for col_btn, (label, preset_instr) in zip(quick_cols, fe_presets):
            if col_btn.button(label, use_container_width=True):
                fe_instructions = preset_instr

        if st.button("🚀 Run Feature Engineering", use_container_width=True, type="primary"):
            if not openai_api_key and "OPENAI_API_KEY" not in os.environ:
                st.error("OpenAI API key required.")
            else:
                from eda_agents.agents.feature_engineering import FeatureEngineeringAgent
                llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key or os.environ["OPENAI_API_KEY"])
                agent = FeatureEngineeringAgent(llm, target_variable=target_var)
                with st.spinner("Running feature engineering pipeline…"):
                    instr = fe_instructions or "Apply default feature engineering steps."
                    from langchain_core.messages import HumanMessage as HM
                    import time as _time
                    _fe_config = {"configurable": {"thread_id": f"fe_{int(_time.time() * 1000)}"}}
                    result = agent.invoke({
                        "messages": [HM(content=instr)],
                        "data_raw": st.session_state["data_raw"],
                        "hitl_enabled": False,
                        "target_variable": target_var,
                    }, config=_fe_config)
                    engineered = result.get("engineered_data")
                    if engineered:
                        df_eng = pd.DataFrame(engineered)
                        n_before = len(df_fe.columns)
                        n_after = len(df_eng.columns)
                        st.success(f"✅ Done! Columns: {n_before} → {n_after} | Rows: {len(df_eng):,}")
                        st.dataframe(df_eng.head(10), use_container_width=True)

                        apply_col, _ = st.columns([1, 3])
                        if apply_col.button("💾 Apply & Update Dataset", type="primary"):
                            st.session_state["data_raw"] = engineered
                            st.session_state["history"].append({
                                "action": "⚙️ Feature Engineering",
                                "description": f"{n_before}→{n_after} cols",
                                "data": engineered,
                            })
                            st.toast("✅ Engineered dataset applied!")
                            st.rerun()

                        if result.get("plan"):
                            with st.expander("📝 Engineering Plan"):
                                st.markdown(result["plan"])
                        if result.get("code"):
                            with st.expander("🧑‍💻 Generated Code"):
                                st.code(result["code"], language="python")
                    else:
                        error = result.get("error", "Unknown error.")
                        st.error(f"Feature engineering failed: {error}")

        st.markdown("---")
        st.markdown("#### 📤 Export Pipeline as Python Script")
        st.write("Download a reproducible Python script that replays your entire session transformation history.")
        script = _export_pipeline_as_python()
        st.download_button(
            "⬇️ Download pipeline.py",
            data=script,
            file_name="pipeline.py",
            mime="text/x-python",
            use_container_width=True,
        )
        with st.expander("Preview script"):
            st.code(script, language="python")


# --- ML Training ---
elif navigation == "🤖 ML Training":
    if st.session_state["data_raw"] is None:
        st.warning("Please upload data first on the Home page.")
    else:
        st.markdown("### 🤖 AutoML Model Training")
        st.write("Train a scikit-learn model with one click. The agent selects the best algorithm, evaluates it, and returns feature importances.")

        df_ml = pd.DataFrame(st.session_state["data_raw"])
        cols_ml = df_ml.columns.tolist()

        with st.container(border=True):
            ml_col1, ml_col2 = st.columns(2)
            with ml_col1:
                ml_target = st.selectbox("Target Column", cols_ml, index=len(cols_ml) - 1)
            with ml_col2:
                ml_task = st.selectbox("Task Type", ["auto (infer)", "classification", "regression"])
                ml_task_val = "auto" if ml_task.startswith("auto") else ml_task

            ml_instructions = st.text_area(
                "Custom Instructions (optional)",
                placeholder="e.g. 'Use a Random Forest with 200 trees' or 'Include cross-validation'",
                height=70,
            )

        with st.container(border=True):
            st.markdown("#### ⚡ Quick Settings")
            q1, q2, q3 = st.columns(3)
            q1.metric("Rows", f"{len(df_ml):,}")
            q2.metric("Features", len(df_ml.columns) - 1)
            q3.metric("Target", ml_target)

        if "ml_session_result" in st.session_state:
            st.markdown("---")
            st.markdown("#### 📊 Last Training Result")
            _render_ml_results(st.session_state["ml_session_result"], key_prefix="ml_page")

        if st.button("🚀 Train Model", use_container_width=True, type="primary"):
            if not openai_api_key and "OPENAI_API_KEY" not in os.environ:
                st.error("OpenAI API key required.")
            else:
                from eda_agents.agents.sklearn_ml import SklearnMLAgent
                llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key or os.environ["OPENAI_API_KEY"])
                agent = SklearnMLAgent(llm, target_variable=ml_target, task_type=ml_task_val)
                with st.spinner("Training model… (this may take 30–90 seconds)"):
                    instr = ml_instructions or f"Train a model to predict '{ml_target}'."
                    from langchain_core.messages import HumanMessage as HM
                    result = agent.invoke({
                        "messages": [HM(content=instr)],
                        "data_raw": st.session_state["data_raw"],
                        "hitl_enabled": False,
                        "target_variable": ml_target,
                        "task_type": ml_task_val,
                    })
                    ml_results = result.get("ml_results", {})
                    if ml_results:
                        st.session_state["ml_session_result"] = ml_results
                        st.rerun()
                    else:
                        err = result.get("error", "Training failed.")
                        st.error(f"❌ {err}")
                        if result.get("code"):
                            with st.expander("Debug: generated code"):
                                st.code(result["code"], language="python")


# --- Data Loader ---
elif navigation == "📁 Data Loader":
    st.markdown("### 📁 AI Data Loader")
    st.write("Browse your file system and load datasets using natural language. The agent can inspect, preview, and load any CSV, Excel, Parquet, or JSON file.")

    with st.container(border=True):
        st.markdown("#### Quick Actions")
        dl_cols = st.columns(3)
        dl_presets = [
            ("📂 Browse Current Dir", "List all data files in the current directory."),
            ("🔎 Find CSVs", "Search for all CSV files in the current directory and subdirectories."),
            ("📄 Show File Info", "List the data files here and give me info on the first one you find."),
        ]
        for col_btn, (label, preset_q) in zip(dl_cols, dl_presets):
            if col_btn.button(label, use_container_width=True):
                st.session_state["dl_query"] = preset_q

    st.markdown("---")
    dl_query = st.text_input(
        "Ask the Data Loader Agent",
        value=st.session_state.pop("dl_query", ""),
        placeholder="e.g. 'Find CSV files in ~/Downloads and show me what columns they have'",
    )

    if st.button("🔍 Search & Load", use_container_width=True) and dl_query:
        if not openai_api_key and "OPENAI_API_KEY" not in os.environ:
            st.error("OpenAI API key required.")
        else:
            from eda_agents.agents.data_loader import DataLoaderToolsAgent
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key or os.environ["OPENAI_API_KEY"])
            agent = DataLoaderToolsAgent(llm)
            with st.spinner("Searching…"):
                agent.invoke_agent(dl_query)
            st.markdown("#### Agent Response")
            st.markdown(agent.get_ai_message())
            if agent.get_tool_calls():
                st.caption("Tools used: " + " → ".join(f"`{t}`" for t in agent.get_tool_calls()))

    st.markdown("---")
    st.markdown("#### 📥 Manual File Upload")
    st.write("Or upload a file directly:")
    uploaded = st.file_uploader("Upload CSV / Excel / JSON", type=["csv", "xlsx", "xls", "json"])
    if uploaded:
        try:
            ext = uploaded.name.rsplit(".", 1)[-1].lower()
            if ext == "csv":
                df_ul = pd.read_csv(uploaded)
            elif ext in ("xlsx", "xls"):
                df_ul = pd.read_excel(uploaded)
            elif ext == "json":
                df_ul = pd.read_json(uploaded)
            else:
                st.error(f"Unsupported format: {ext}")
                df_ul = None

            if df_ul is not None:
                st.success(f"✅ Loaded {uploaded.name}: {df_ul.shape[0]:,} rows × {df_ul.shape[1]} columns")
                st.dataframe(df_ul.head(10), use_container_width=True)
                if st.button("💾 Use this as active dataset", type="primary"):
                    records = df_ul.to_dict(orient="records")
                    st.session_state["data_raw"] = records
                    st.session_state["history"] = [{
                        "action": f"📂 {uploaded.name}",
                        "description": f"{len(df_ul):,} rows",
                        "data": records,
                    }]
                    st.toast(f"✅ {uploaded.name} loaded as active dataset!")
                    st.session_state["navigation_radio"] = "🏠 Home"
                    st.rerun()
        except Exception as e:
            st.error(f"Could not load file: {e}")
