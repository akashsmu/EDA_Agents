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
- **Autonomous Report Generation**: Combines LLM-driven insights with automated visualization reports (via Sweetviz).
- **Sandboxed Execution**: Ensures that LLM-generated code is executed safely before results are returned to the user.
- **Human-in-the-Loop (HITL)**: Support for user approval on suggested cleaning plans before execution.
- **Agent Transparency**: Real-time display of the multi-agent thinking process in the UI using LangGraph streams.
- **Factory Pattern**: Standardized agent instantiation via a centralized factory to cleanly manage multiple specialized workers.

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



