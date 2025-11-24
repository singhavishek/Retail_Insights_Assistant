import pandas as pd
import os
import glob

def load_data(file_path):
    """
    Loads a CSV file into a Pandas DataFrame.
    """
    try:
        # Attempt to read with default settings
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Fallback for encoding issues if necessary
        try:
            df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
            return df
        except Exception as e2:
             print(f"Error loading {file_path} with latin1: {e2}")
             return None

def preprocess_data(df):
    """
    Preprocesses the dataset based on its columns to handle Amazon, International, or Stock reports.
    """
    if df is None:
        return None
    
    # Copy to avoid SettingWithCopy warnings
    df = df.copy()

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # --- Amazon Sales Report ---
    if 'Order ID' in df.columns:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        numeric_cols = ['Qty', 'Amount', 'ship-postal-code']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Amount' in df.columns:
            df['Amount'] = df['Amount'].fillna(0.0)

    # --- International Sales Report ---
    elif 'GROSS AMT' in df.columns:
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        
        numeric_cols = ['PCS', 'RATE', 'GROSS AMT']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Sale Report (Stock) ---
    elif 'Stock' in df.columns:
        df['Stock'] = pd.to_numeric(df['Stock'], errors='coerce')

    # --- Catalog / Pricing (May-2022, P L March 2021) ---
    elif 'Style Id' in df.columns:
        # Convert all MRP/TP columns to numeric
        cols_to_numeric = [c for c in df.columns if 'MRP' in c or 'TP' in c or 'Weight' in c]
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Expense Report ---
    elif 'Expance' in df.columns or 'Recived Amount' in df.columns:
        # This file seems to have a double header or unstructured format.
        # We'll try to clean it up if row 0 looks like a header (Particular, Amount)
        if not df.empty and 'Particular' in df.iloc[0].values:
            # Promote first row to header
            new_header = df.iloc[0]
            df = df[1:]
            df.columns = new_header
            # Reset index
            df = df.reset_index(drop=True)
            
        # Convert Amount columns
        for col in df.columns:
            if 'Amount' in str(col):
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Warehouse Comparison ---
    elif 'Shiprocket' in df.columns:
        # Similar issue with header potentially on row 1
        if not df.empty and 'Heads' in df.iloc[0].values:
             new_header = df.iloc[0]
             df = df[1:]
             df.columns = new_header
             df = df.reset_index(drop=True)
             
             # Deduplicate columns if any
             df.columns = pd.io.parsers.ParserBase({'names':df.columns})._maybe_dedup_names(df.columns)

    return df

def load_all_data(data_dir):
    """
    Loads all CSV files from the directory into a dictionary of DataFrames.
    """
    dataframes = {}
    if not os.path.exists(data_dir):
        return dataframes
        
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    for file_path in files:
        file_name = os.path.basename(file_path).replace(".csv", "").replace(" ", "_").replace("-", "_").lower()
        try:
            df = pd.read_csv(file_path, low_memory=False)
            df = preprocess_data(df)
            if df is not None:
                dataframes[file_name] = df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return dataframes

def get_dataset_summary(data_input):
    """
    Returns a string summary of the dataset(s) for the LLM context.
    Accepts either a single DataFrame or a dictionary of DataFrames.
    """
    if data_input is None:
        return "No data loaded."
        
    summary = []
    
    if isinstance(data_input, pd.DataFrame):
        dfs = {"default_df": data_input}
    elif isinstance(data_input, dict):
        dfs = data_input
    else:
        return "Invalid data format."
        
    for name, df in dfs.items():
        summary.append(f"\n--- DataFrame: {name} ---")
        summary.append(f"Total Rows: {len(df)}")
        summary.append(f"Columns: {', '.join(df.columns)}")
        
        # Add column types and sample values (limit to first 5 cols to save tokens if many dfs)
        for col in df.columns[:10]: 
            dtype = df[col].dtype
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
            summary.append(f"- {col} ({dtype}): e.g., {sample}")
            
    return "\n".join(summary)
