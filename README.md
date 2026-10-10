# EDA Agents

An intelligent, multi-agent framework designed to autonomously conduct Exploratory Data Analysis (EDA). This project leverages large language models (LLMs) and agentic orchestration to transform raw data into actionable insights through automated cleaning, visualization, and statistical analysis.

## What Problem Does This Solve?

Exploratory Data Analysis is often a repetitive and time-consuming process for data scientists. Analysts spend significant time:
- Writing boilerplate code for data cleaning (handling nulls, outliers, types).
- Manually generating plots to understand distributions and correlations.
- Synthesizing findings into coherent reports.

**EDA Agents** automates these workflows, allowing users to interact with their data using natural language. It acts as an autonomous data partner that can explore datasets, suggest cleaning steps, and generate sophisticated visualizations without manual coding.

## Technologies Used

- **Core Framework**: LangChain & LangGraph for agent orchestration and multi-agent workflows.
- **Large Language Models**: Powered by OpenAI's GPT-4 (via LangChain integrations).
- **Interface**: Streamlit for a responsive, interactive web application.
- **Data Processing**: Pandas, OpenPyXL.
- **Visualization**: Plotly (interactive), Matplotlib, Seaborn, Plotnine.
- **Automated EDA Tools**: Sweetviz, Missingno, PyTimeTK (for time series).
- **Security**: Sandboxed code execution in secure subprocesses.
- **DevOps/Testing**: Docker & Docker Compose for orchestration, Pytest for agent reliability, pip for dependency management.

## Key Features

- **Multi-Agent Supervisor System**: A central supervisor agent coordinates specialized sub-agents to solve complex queries.
- **Data Specialist Agents**:
    - **Visualization Agent**: Generates interactive Plotly charts with automatic column profiling and alias resolution.
    - **Wrangling Agent**: Performs complex data transformations and explains the logic in plain English.
    - **Cleaning Agent**: Specifically focused on data quality—identifying and fixing missing values, outliers, and formatting issues.
    - **EDA Tools Agent**: A tool-calling (ReAct) analyst that picks from 11 EDA tools to answer questions without modifying data (see below).
- **Autonomous Report Generation**: Combines LLM-driven insights with automated visualization reports (via Sweetviz).
- **Sandboxed Execution**: Ensures that LLM-generated code is executed safely before results are returned to the user.
- **Human-in-the-Loop (HITL)**: Support for user approval on suggested cleaning plans before execution.
- **Agent Transparency**: Real-time display of the multi-agent thinking process in the UI using LangGraph streams.
- **Factory Pattern**: Standardized agent instantiation via a centralized factory to cleanly manage multiple specialized workers.

## EDA Tools Agent

`EDAToolsAgent` ([src/eda_agents/agents/eda_tools.py](src/eda_agents/agents/eda_tools.py)) is modelled on
[ai-data-science-team's EDA Tools Agent](https://github.com/business-science/ai-data-science-team/blob/master/examples/ds_agents/eda_tools_agent.ipynb).
The supervisor routes analytical questions to it ("is X related to Y?", "how clean is this data?"). It can also be used on its own:

```python
from eda_agents.agents.eda_tools import EDAToolsAgent

agent = EDAToolsAgent(llm)
agent.invoke_agent("Does contract type affect churn?", data_raw=df)
agent.get_ai_message()                        # markdown answer
agent.get_tool_calls()                        # ['analyze_relationship']
agent.get_artifacts()["analyze_relationship"] # p-value, effect size, plotly figure...
```

| Tool | What it does |
|---|---|
| `explain_data` / `describe_dataset` | Narrative summary and `describe()` statistics |
| `visualize_missing` | missingno matrix, bar and heatmap plots |
| `generate_correlation_funnel` | Binarized correlation against a target value (pytimetk, or a built-in pandas fallback) |
| `generate_sweetviz_report` / `generate_dtale_report` | Full HTML report / interactive explorer |
| `assess_data_quality` | 0–100 quality score and grade. Flags missing data, duplicates, constant columns, numbers stored as text, mixed types, inconsistent labels and ID-like columns, each with a suggested fix |
| `detect_outliers` | IQR or z-score outliers per column, with a standardized box plot |
| `analyze_correlations` | Pearson/Spearman/Kendall matrix, strongest pairs, multicollinearity warnings |
| `profile_distributions` | Skewness, kurtosis, normality test, shape label and suggested transform |
| `analyze_relationship` | Picks the right test for two columns (correlation, t-test/ANOVA, chi-square) and reports the p-value, effect size and a plain-English conclusion |

Design notes:
- **The data never goes to the LLM.** Tools get the dataset through `InjectedState("data_raw")`. The model sees only the schema (column names, types and category values, added to the system prompt) and the tools' text summaries.
- **Tool failures don't crash the run.** Errors (e.g. a wrong column name) come back to the model as error messages so it can retry, and a model-call limit stops runaway loops.
- **One-click fixes in the UI.** Each issue in a data quality report has a 🛠️ Fix button that sends the suggested fix to the supervisor, which routes it to the cleaning agent.

See [examples/eda_tools_agent.ipynb](examples/eda_tools_agent.ipynb) for a walkthrough. The Home page also has a **sample churn dataset** with built-in quality issues to try the tools on.

## Outcome & Results

The project results in a production-ready EDA tool that:
- Reduces the time from "raw data" to "insights" by automating the coding of plots and transformations.
- Provides a conversational interface where non-technical users can perform complex data analysis.
- Generates high-quality, shareable HTML reports (via Sweetviz integration) automatically.
- Ensures reproducible results through structured logging and sandboxed code execution.

## Why This Matters

In a data-driven world, the ability to rapidly understand new datasets is a competitive advantage. **EDA Agents** democratizes data analysis by lowering the technical barrier to entry. It ensures that data exploration is thorough, consistent, and fast—reducing the risk of missing critical patterns due to the manual effort of exploratory coding.

## File Structure

```text
EDA_Agents/
├── src/eda_agents/
│   ├── agents/          # Agent logic (Visualization, Wrangling, Cleaning, etc.)
│   │   ├── base.py      # Base class for all agents
│   │   ├── cleaning.py  # Data cleaning specialization
│   │   ├── supervisor.py # Central orchestrator
│   │   └── graph.py     # LangGraph state and workflow definition
│   ├── multiagents/     # Sequential & chained multi-agent implementations
│   ├── tools/           # Custom tools used by agents (EDA, DataFrame operations)
│   ├── ui/              # Streamlit frontend components
│   ├── utils/           # Sandbox execution, logging, and plotting utilities
│   ├── templates/       # Common graph and prompt templates
│   └── models/          # LLM configuration and factory
├── tests/               # Comprehensive test suite (agent logic, tools)
├── config/              # Application configuration
├── data/                # Sample datasets (if any)
├── reports/             # Generated EDA output files
├── Dockerfile           # Docker image configuration instructions
├── docker-compose.yml   # Docker Compose orchestration configuration
└── README.md            # You are here
```

## Installation

### 1. Prerequisites
- Python 3.10+
- OpenAI API Key

### 2. Setup
```bash
git clone <repository-url>
cd EDA_Agents

# Local Install
python3 -m venv eda
source eda/bin/activate
pip install -r requirements.txt

# Or run via Docker
docker-compose up --build
```

### 3. Environment
Create a `.env` file:
```bash
OPENAI_API_KEY=your_key_here
```

## Usage

### Run Application
```bash
streamlit run src/eda_agents/ui/app.py
```

### Run Tests
```bash
pytest tests/
```

## Future Work & Contributions

- Support for more data formats (SQL, Parquet).
- Integration with local LLMs (Ollama/LlamaCpp) for privacy.
- Advanced Predictive EDA (automated regression/classification baselines).
- Exporting agent-generated code as notebook snippets.


