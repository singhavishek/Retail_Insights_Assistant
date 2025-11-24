from typing import TypedDict, Annotated, Sequence, Any
import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from src.utils import get_llm
from src.data_loader import get_dataset_summary
import operator

# Define the state of the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    dataframes: dict # Dictionary of DataFrames
    dataset_summary: str
    code: str
    result: str
    error: str
    figure: Any # Matplotlib figure object

# --- Agent Functions ---

def query_resolution_agent(state: AgentState):
    """
    Translates the user's natural language query into Python/Pandas code.
    """
    print("--- Query Resolution Agent ---")
    messages = state['messages']
    summary = state['dataset_summary']
    llm = get_llm()
    
    if not llm:
        return {"error": "LLM not initialized. Check API Key."}

    # Get the latest user question
    user_query = messages[-1].content

    prompt = f"""
    You are a Data Science Assistant. You have multiple pandas DataFrames loaded in memory.
    
    Dataset Summary:
    {summary}
    
    User Query: {user_query}
    
    Your task is to generate Python code using Pandas/Matplotlib/Seaborn to answer the query.
    - The DataFrames are available in a dictionary named `dfs`.
    - You can access them like `dfs['amazon_sale_report']`.
    - Do NOT load the data again.
    - Store the final text/numerical result in a variable named `final_result`.
    - If the user asks for a plot/graph/chart:
        - Create the plot using matplotlib or seaborn.
        - **IMPORTANT**: Assign the figure object to a variable named `fig`.
        - Example: `fig = plt.gcf()`
        - Do NOT use `plt.show()`.
    - Wrap your code in a markdown block like:
    ```python
    ...
    ```
    - Do not include any explanations, just the code.
    """
    
    response = llm.invoke(prompt)
    content = response.content
    
    # Extract code from markdown block
    code = ""
    if "```python" in content:
        code = content.split("```python")[1].split("```")[0].strip()
    elif "```" in content:
        code = content.split("```")[1].split("```")[0].strip()
    else:
        code = content.strip()
        
    return {"code": code}

def data_extraction_agent(state: AgentState):
    """
    Executes the generated Python code.
    """
    print("--- Data Extraction Agent ---")
    code = state['code']
    dataframes = state['dataframes']
    
    if not code:
        return {"error": "No code generated."}
    
    # Setup plotting context
    plt.clf() # Clear existing plots
    
    local_scope = {'dfs': dataframes, 'pd': pd, 'plt': plt}
    
    try:
        exec(code, {}, local_scope)
        result = local_scope.get('final_result', "Done.")
        fig = local_scope.get('fig', None)
        
        return {"result": str(result), "figure": fig, "error": None}
    except Exception as e:
        return {"error": str(e), "result": None}

def validation_agent(state: AgentState):
    """
    Validates the result and checks for errors.
    """
    print("--- Validation Agent ---")
    error = state.get('error')
    
    if error:
        return {"messages": [AIMessage(content=f"Error executing code: {error}")]}
    
    # Simple validation: if we have a result, we assume it's good for now.
    # In a real system, we might double check constraints.
    return {}

def response_agent(state: AgentState):
    """
    Formats the final answer for the user.
    """
    print("--- Response Agent ---")
    result = state.get('result')
    error = state.get('error')
    messages = state['messages']
    user_query = messages[-1].content
    llm = get_llm()

    if error:
         return {"messages": [AIMessage(content=f"I encountered an error while analyzing the data: {error}")]}

    prompt = f"""
    You are a helpful Retail Insights Assistant.
    
    User Query: {user_query}
    
    Analysis Result:
    {result}
    
    Please provide a natural language answer to the user based on the analysis result.
    Be concise and professional.
    """
    
    response = llm.invoke(prompt)
    return {"messages": [response]}

# --- Graph Construction ---

def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("query_resolution", query_resolution_agent)
    workflow.add_node("data_extraction", data_extraction_agent)
    workflow.add_node("validation", validation_agent)
    workflow.add_node("response", response_agent)
    
    workflow.set_entry_point("query_resolution")
    
    workflow.add_edge("query_resolution", "data_extraction")
    workflow.add_edge("data_extraction", "validation")
    
    # Conditional edge based on error
    def check_error(state):
        if state.get('error'):
            return "response" # Go straight to response to report error
        return "response"

    workflow.add_edge("validation", "response")
    workflow.add_edge("response", END)
    
    app = workflow.compile()
    return app
