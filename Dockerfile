# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies via pip (as specified in memory.md)
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user (appuser) and set permissions
RUN groupadd -r appuser && useradd -mr -g appuser appuser \
    && chown -R appuser:appuser /app

# Switch to the non-root user for improved security
USER appuser

# Expose the standard Streamlit port
EXPOSE 8501

# Add a health check to verify Streamlit is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Define the default command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "src/eda_agents/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
