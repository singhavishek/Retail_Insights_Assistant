import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_llm():
    """
    Initializes and returns the Groq LLM.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Warning: GROQ_API_KEY not found in environment.")
        return None
    
    # Using Llama3-70b-8192 for high performance reasoning
    llm = ChatGroq(model="openai/gpt-oss-20b", api_key=api_key, temperature=0)
    return llm
