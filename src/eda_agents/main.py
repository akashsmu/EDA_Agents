import os
import sys

# Add the src directory to the system path to ensure modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def main():
    """
    Entry point for the application.
    Currently, this acts as a placeholder or could be used to launch the streamlit app programmatically.
    """
    print("EDA Agents Framework Initialized.")
    print("To run the UI, use: poetry run streamlit run src/eda_agents/ui/app.py")

if __name__ == "__main__":
    main()
