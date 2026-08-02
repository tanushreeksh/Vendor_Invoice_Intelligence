import pandas as pd
import sqlite3

from sklearn.model_selection import train_test_split


def load_invoice_data(db_path: str = "data/inventory.db"):
    conn = sqlite3.connect(db_path)

    query = """
    WITH purchase_agg AS(
        SELECT 
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG (julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay 
        FROM purchases p
        GROUP BY PONumber  
        )
        SELECT 
            vi.Quantity AS invoice_quantity,
            vi.Dollars AS invoice_dollars,
            vi.Freight,
            julianday(vi.InvoiceDate) - julianday(vi.PODate) AS days_po_to_invoice,
            julianday(vi.PayDate) - julianday(vi.InvoiceDate) AS days_to_pay,
            pa.total_brands,
            pa.total_item_quantity,
            pa.total_item_dollars,
            pa.avg_receiving_delay
        FROM vendor_invoice vi
    
        LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """

    master_df = pd.read_sql_query(query, conn)
    conn.close()
    return master_df



def create_invoice_risk_label(row):
    # No matching purchase order found (LEFT JOIN produced NaNs) -
    # treat as risky rather than silently defaulting to "not risky"
    if pd.isna(row['total_item_dollars']) or pd.isna(row['avg_receiving_delay']):
        return 1

    # Invoice total mismatch with item-level total
    if (abs(row['invoice_dollars'] - row['total_item_dollars']) > 5):
        return 1
    
    # Abnormally high receiving delay
    if row['avg_receiving_delay'] > 10:
        return 1
    
    return 0



def apply_labels(master_df):
    master_df['flag_invoice'] = master_df.apply(create_invoice_risk_label, axis = 1)
    return master_df



def split_data(master_df, features, target):
    X = master_df[features]
    y = master_df[target]

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )