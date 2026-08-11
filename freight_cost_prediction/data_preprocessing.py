import pandas as pd
import sqlite3

from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path: str = "data/inventory.db") -> pd.DataFrame:
    """
    Load vendor invoice data from SQLite database
    """

    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM vendor_invoice"

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build date-derived, ratio, and vendor-scale features on top of the
    raw invoice columns.
    """

    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["PODate"] = pd.to_datetime(df["PODate"])
    df["PayDate"] = pd.to_datetime(df["PayDate"])

    # Date-derived features
    df["days_po_to_invoice"] = (df["InvoiceDate"] - df["PODate"]).dt.days
    df["days_to_pay"] = (df["PayDate"] - df["InvoiceDate"]).dt.days
    df["invoice_month"] = df["InvoiceDate"].dt.month

    # Ratio feature - separates expensive item, few units from cheap item, bulk order
    df["dollars_per_unit"] = df["Dollars"] / df["Quantity"].replace(0, pd.NA)

    # Vendor-scale proxy: how many invoices this vendor has in the dataset.
    # Avoids one-hot encoding a high-cardinality categorical column.
    vendor_counts = df["VendorNumber"].value_counts()
    df["vendor_invoice_count"] = df["VendorNumber"].map(vendor_counts)

    return df



def prepare_features(df: pd.DataFrame):
    """
    Select features and target variable
    """

    features = [
        "Quantity",
        "Dollars",
    ]

    df = df.dropna(subset=features + ["Freight"])

    X = df[features]
    y = df["Freight"]

    return X, y



def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """

    return train_test_split(
        X, y, test_size = test_size, random_state=random_state
    )