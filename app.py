import streamlit as st
import pandas as pd
import os
import glob
from dotenv import load_dotenv
from src.data_loader import load_all_data, get_dataset_summary
from src.agents import create_graph
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="Retail Insights Assistant", layout="wide")

st.title("🛍️ Retail Insights Assistant")
st.markdown("GenAI-powered analysis for your sales data.")

# Sidebar for Setup
with st.sidebar:
    st.header("Configuration")
    
    # API Key Handling
    if os.getenv("GROQ_API_KEY"):
        st.success("✅ Groq API Key detected from environment.")
    else:
        api_key = st.text_input("Enter Groq API Key", type="password")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            st.success("API Key set!")
        else:
            st.warning("Please provide an API Key to proceed.")
    
    st.divider()
    
    st.info("📂 All datasets in the 'Data' folder are loaded automatically.")
    
    with st.expander("❓ Sample Questions"):
        st.markdown("""
        **General**
        - What is the total revenue?
        - Count of orders by Status?
        
        **Time**
        - Total sales in April 2022?
        - Plot daily sales trend.
        
        **Product/Geo**
        - Top 5 selling categories?
        - Sales by Ship State?
        
        **Specialized (Catalog/Expense)**
        - Avg TP for Kurta?
        - Total Expense Amount?
        - Compare Amazon vs Flipkart MRP.
        """)

    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = None
        st.session_state.dataset_summary = None

# Load Data Logic
if st.session_state.dataframes is None:
    with st.spinner("Loading all datasets..."):
        try:
            data_dir = "Data"
            dfs = load_all_data(data_dir)
            
            if dfs:
                st.session_state.dataframes = dfs
                st.session_state.dataset_summary = get_dataset_summary(dfs)
                st.success(f"Loaded {len(dfs)} datasets successfully!")
                with st.expander("View Loaded Datasets"):
                    for name, df in dfs.items():
                        st.write(f"**{name}** ({len(df)} rows)")
                        st.dataframe(df.head(3))
            else:
                st.error("No CSV files found in 'Data' folder.")
                
        except Exception as e:
            st.error(f"Error loading files: {e}")

# Main Chat Interface
if st.session_state.dataframes is not None:
    
    # Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask a question about your sales data..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with Agents
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    app = create_graph()
                    inputs = {
                        "messages": [HumanMessage(content=prompt)],
                        "dataframes": st.session_state.dataframes,
                        "dataset_summary": st.session_state.dataset_summary
                    }
                    
                    # Run the graph
                    result = app.invoke(inputs)
                    response_content = result["messages"][-1].content
                    
                    st.markdown(response_content)
                    
                    # Display plot if available
                    if "figure" in result and result["figure"]:
                        st.pyplot(result["figure"])
                    
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
