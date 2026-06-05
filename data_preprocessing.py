import pandas as pd

def load_and_clean_data(csv_path):
    """
    Loads raw e-commerce data and performs basic cleaning
    """
    # Read CSV
    df = pd.read_csv(csv_path, encoding="cp1252")

    # Drop missing values
    df.dropna(inplace=True)

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(
        df['InvoiceDate'],
        format='%m/%d/%Y %H:%M'
    )

    # Create revenue columns
    df['TotalRevenue'] = df['Quantity'] * df['UnitPrice']
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    return df
