# EDA Agents

An intelligent, agentic framework for Exploratory Data Analysis (EDA) powered by LangChain, LangGraph, and Streamlit. This project uses a multi-agent architecture to autonomously visualize, clean, and analyze your data.

## 🚀 Features

- **Multi-Agent Architecture**:
    - **Visualization Agent**: Generates Plotly charts based on natural language queries, featuring **advanced column profiling**, **automated alias resolution**, and **robust fallback chart generation**.
    - **Wrangling Agent**: Cleans and transforms data using Pandas, equipped with enhanced methods for **data manipulation, in-depth analysis**, and **conversational chat functionality**.
    - **Orchestrator**: Routes user requests to the appropriate specialist agent.
- **Advanced EDA Tools**:
    - **Custom Report Generation**: Autonomously generate extensive EDA reports combining intelligent text summaries and visualizations.
    - **Sweetviz**: Comprehensive automated EDA reports.
    - **Missingno**: Visual analysis of missing data.
- **Robustness & Security**:
    - **Comprehensive Testing**: Full test suite powered by `pytest` ensuring stable agent behavior and prompt regression prevention.
    - **Sandboxed Execution**: Generated code is executed in a secure subprocess.
- **Interactive UI**: User-friendly Streamlit interface for chat, tools, and reports.

## 🛠️ Installation

### Prerequisites
- **Python 3.10+** (check with `python3 --version`)

### Quick Start (Recommended)

```bash
git clone <repository-url>
cd EDA_Agents
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ensure Setup in Environment

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

## 🏃 Usage

### Running the App

```bash
source .venv/bin/activate
streamlit run src/eda_agents/ui/app.py
```

### Running Tests

We use `pytest` for ensuring application stability and agent reliability.

```bash
pytest tests/
```

### Workflow
1.  **Upload Data**: Upload a CSV file in the sidebar.
2.  **Chat & Plot**: Ask questions like "Plot a scatter of Age vs Fare" or "Drop rows with missing Age".
3.  **EDA Tools**: Use the "Quick EDA Tools" tab for instant summaries and missing value plots.
4.  **Reports**: Generate full HTML reports with Sweetviz.

## 🏗️ Architecture

- `src/eda_agents/agents/`: Agent logic (`visualization.py`, `wrangling.py`, `base.py`) and graph orchestrator (`graph.py`).
- `src/eda_agents/tools/`: Specialized tools for EDA (`eda.py`, `dataframe.py`).
- `src/eda_agents/utils/`: Utility functions for sandboxed execution and plotting.
- `src/eda_agents/ui/`: Streamlit application code.

