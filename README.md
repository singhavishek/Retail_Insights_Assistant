# Retail Insights Assistant

A GenAI-powered application to analyze retail sales data using a Multi-Agent architecture.

## Features
- **Data Ingestion**: Put CSV data file in the `Data/` directory inside root.
- **Natural Language Querying**: Ask questions like "What was the total revenue in April?"
- **Multi-Agent Backend**: Uses LangGraph to orchestrate query translation, data extraction, and response generation.
- **Scalable Design**: Built with modular components.

## Setup

1. **Prerequisites**: Python 3.9+
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **API Key**:
   - You need a **Groq API Key**.
   - You can enter it in the UI or set it in a `.env` file as `GROQ_API_KEY=your_key`.
4. **Data Files**:
   - Place your CSV files in the `Data/` directory.

## Sample Questions
Here are some questions you can ask the assistant:

**📊 General Statistics**
- "What is the total revenue?"
- "How many total orders are there?"
- "What is the average order value?"

**🏆 Product & Category Analysis**
- "What are the top 5 selling categories?"
- "Which product (SKU) has the highest sales?"
- "What is the sales distribution by Size?"

**📅 Time-Series Analysis**
- "What was the total revenue in April 2022?"
- "Which month had the highest sales?"
- "Show me the daily sales trend."

**📍 Geographical & Operational**
- "Which state has the highest number of orders?"
- "What are the top 3 cities by revenue?"
- "How many orders were Cancelled?"
- "What is the split between Merchant and Amazon fulfilment?"

**📈 Visualizations**
- "Plot the sales by category."
- "Show a bar chart of sales by state."
- "Plot the daily sales trend."

**📂 Specialized Datasets**
*For Catalog/Pricing (May-2022, P&L):*
- "What is the average TP (Transfer Price) for Kurta?"
- "Compare Amazon MRP vs Flipkart MRP for Style Os206_3141."
- "Which category has the highest average weight?"

*For Expense Reports:*
- "What is the total expense amount?"
- "List the top 3 expense categories."

*For Warehouse Comparison:*
- "Compare the rates for Inbound processing."

## Running the App

```bash
streamlit run app.py
```

## Architecture
The system uses a graph-based agent workflow:
1. **Query Resolution Agent**: Converts user text to Python/Pandas code.
2. **Data Extraction Agent**: Executes the code on the loaded DataFrame.
3. **Validation Agent**: Checks for execution errors.
4. **Response Agent**: Formats the result into a natural language answer.


Structure:

root/
│
├── Data/                   # Include the sample CSVs here
│   ├── Amazon Sale Report.csv
│   └── ...
│
├── src/                    # Source code module
│   ├── agents.py
│   ├── data_loader.py
│   └── utils.py
│
├── app.py                  # Main Streamlit Entry point
├── requirements.txt        # Dependencies
├── .env.example            # Template for API keys (create your own .env)
├── README.md               # Instructions
└── Presentation/           # The presentation (Exported from PPT)
    ├── Retail Insights Assistant .pdf