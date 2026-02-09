# EDA Agents

An intelligent, agentic framework for Exploratory Data Analysis (EDA) powered by LangChain, LangGraph, and Streamlit. This project uses a multi-agent architecture to autonomously visualize, clean, and analyze your data.

## 🚀 Features

- **Multi-Agent Architecture**:
    - **Visualization Agent**: Generates Plotly charts based on your natural language queries.
    - **Wrangling Agent**: Cleans and transforms data using Pandas.
    - **Orchestrator**: Routes user requests to the appropriate specialist agent.
- **Advanced EDA Tools**:
    - **Sweetviz**: Comprehensive automated EDA reports.
    - **Missingno**: Visual analysis of missing data.
    - **Correlation Funnel**: Identify drivers of target variables.
    - **D-Tale**: Interactive data exploration.
- **Sandboxed Execution**: Generated code is executed in a secure subprocess.
- **Interactive UI**: User-friendly Streamlit interface for chat, tools, and reports.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd EDA_Agents
    ```

2.  **Install dependencies using Poetry**:
    ```bash
    poetry install
    ```

3.  **Set up environment variables**:
    Create a `.env` file and add your OpenAI API key:
    ```bash
    OPENAI_API_KEY=sk-...
    ```

## 🏃 Usage

Run the Streamlit application:

```bash
poetry run streamlit run src/eda_agents/ui/app.py
```

### Workflow
1.  **Upload Data**: Upload a CSV file in the sidebar.
2.  **Chat & Plot**: Ask questions like "Plot a scatter of Age vs Fare" or "Drop rows with missing Age".
3.  **EDA Tools**: Use the "Quick EDA Tools" tab for instant summaries, missing value plots, and correlation analysis.
4.  **Reports**: Generate full HTML reports with Sweetviz or launch D-Tale for deep diving.

## 🏗️ Architecture

- `src/eda_agents/agents/`: Contains the agent logic (`visualization.py`, `wrangling.py`, `base.py`) and the graph orchestrator (`graph.py`).
- `src/eda_agents/tools/`: Specialized tools for EDA (`eda.py`, `dataframe.py`).
- `src/eda_agents/utils/`: Utility functions for sandboxed execution and plotting.
- `src/eda_agents/ui/`: Streamlit application code.

## 🤝 Contributing

Contributions are welcome! Please submit a Pull Request.
