# EDA Agents Framework

A comprehensive, agentic framework for data ingestion and exploratory data analysis (EDA) using Streamlit, LangChain, and LangGraph.

## Features

- **Data Ingestion**: Support for CSV, Excel, and TXT files.
- **Interactive UI**: Built with Streamlit for a seamless user experience.
- **Agentic Analysis**: (Upcoming) Automated data analysis using LangGraph agents.
- **Reporting**: (Upcoming) Generate downloadable reports.

## Project Structure

```text
EDA_Agents/
├── pyproject.toml          # Poetry configuration and dependencies
├── config/                 # Configuration files
├── src/
│   └── eda_agents/         # Main package
│       ├── ui/             # Streamlit Application
│       ├── agents/         # LangGraph/LangChain Agents
│       ├── tools/          # Custom Tools
│       └── utils/          # Helper functions
└── tests/                  # Unit and integration tests
```

## Getting Started

### Prerequisites

- Python 3.10+
- Poetry

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd EDA_Agents
    ```

2.  Install dependencies:
    ```bash
    poetry install
    ```

3.  Set up environment variables:
    ```bash
    cp .env.example .env
    # Edit .env and add your OPENAI_API_KEY
    ```

### Running the Application

To start the Streamlit UI:

```bash
poetry run streamlit run src/eda_agents/ui/app.py
```

## Development

- **Adding Dependencies**: `poetry add <package>`
- **Running Tests**: `poetry run pytest`

