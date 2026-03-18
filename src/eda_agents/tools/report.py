import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from eda_agents.tools.eda import (
    explain_data, 
    describe_dataset, 
    visualize_missing, 
    generate_correlation_funnel
)
from eda_agents.utils.logger import logger

def generate_custom_report(data_raw: list[dict], wrangling_methods: list[str], analysis_methods: list[str], target_col: str, instructions: str, llm):
    """
    Generates a custom report by:
    1. Applying requested wrangling methods to the data.
    2. Running requested analysis tools to gather insights and plots.
    3. Formatting the text outputs and passing to an LLM to generate an executive summary.
    """
    logger.info("Generating Custom Report")
    df = pd.DataFrame(data_raw)
    
    # 1. Wrangling
    logger.info(f"Applying wrangling methods: {wrangling_methods}")
    for method in wrangling_methods:
        if method == "Drop Duplicates":
            df = df.drop_duplicates()
        elif method == "Drop Missing Values":
            df = df.dropna()
        elif method == "Fill Missing Values (Forward Fill)":
            df = df.fillna(method="ffill")
        elif method == "Fill Missing Values (Backward Fill)":
            df = df.fillna(method="bfill")
    
    cleaned_data_raw = df.to_dict(orient="records")
    
    # 2. Analysis
    logger.info(f"Running analysis methods: {analysis_methods}")
    compiled_texts = []
    artifacts = {}
    
    if "Data Summary" in analysis_methods:
        txt = explain_data.invoke({"data_raw": cleaned_data_raw})
        compiled_texts.append(f"Data Summary:\n{txt}")
        
    if "Descriptive Statistics" in analysis_methods:
        txt, art = describe_dataset.func(data_raw=cleaned_data_raw)
        # Adding some stringified dataframe representation for the LLM
        stats_df_str = pd.DataFrame(art['describe_df']).to_string()
        compiled_texts.append(f"Descriptive Statistics:\n{txt}\n{stats_df_str}")
        artifacts["descriptive_statistics"] = art
        
    if "Missing Values Plot" in analysis_methods:
        txt, art = visualize_missing.func(data_raw=cleaned_data_raw)
        compiled_texts.append(f"Missing Values Analysis:\n{txt}")
        artifacts["missing_plots"] = art
        
    if "Correlation Funnel" in analysis_methods and target_col:
        try:
            txt, art = generate_correlation_funnel.func(data_raw=cleaned_data_raw, target=target_col)
            compiled_texts.append(f"Correlation Funnel (Target: {target_col}):\n{txt}")
            artifacts["correlation_funnel"] = art
        except Exception as e:
            logger.error(f"Failed to generate correlation funnel: {e}")
            compiled_texts.append(f"Correlation Funnel Error: {e}")
    
    # 3. Text Summary with LLM
    logger.info("Synthesizing results with LLM")
    context_text = "\n\n".join(compiled_texts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Data Analyst. I will provide you with the raw output of several data analysis operations run on a dataset. Based on these outputs, write a comprehensive, professional Executive Summary for this dataset. Summarize the shape, quality, key statistics, and any interesting findings. It should be easily readable for stakeholders. If any instructions were provided by the user, ensure you address them. Use markdown formatting. Do not output raw code or JSON, only the final text report."),
        ("user", "User Instructions/Context: {instructions}\n\nAnalysis Results:\n{context}")
    ])
    
    chain = prompt | llm
    try:
        response = chain.invoke({"instructions": instructions, "context": context_text})
        report_summary = response.content
    except Exception as e:
        logger.error(f"LLM Summary generation failed: {e}")
        report_summary = f"Failed to generate textual summary. Error: {e}"
        
    logger.info("Custom report generation completed.")
    return {
        "cleaned_data": cleaned_data_raw,
        "summary_markdown": report_summary,
        "artifacts": artifacts
    }
