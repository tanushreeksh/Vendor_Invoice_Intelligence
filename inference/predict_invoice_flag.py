import joblib
import pandas as pd


MODEL_PATH = "models/flag_invoice.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained invoice risk classifier model
    """

    model = joblib.load(model_path)
    return model



def predict_invoice_flag(input_data):
    """
    Predict risk flag for new vendor invoices

    Parameters:
    input_data : dict

    Returns:
    pd.DataFrame with predicted invoice flag (0 = not risky, 1 = risky)
    """

    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Invoice_Flag'] = model.predict(input_df)
    return input_df



if __name__ == "__main__":
    # Example inference run - real invoice rows from the dataset
    sample_data = {
        "invoice_quantity": [743, 230, 8884, 6, 649],
        "Freight": [59.54, 7.41, 350.52, 1.11, 38.06],
        "total_brands": [3, 7, 26, 93, 38],
        "total_item_quantity": [743, 230, 8884, 5102, 4258],
        "days_po_to_invoice": [17, 21, 17, 16, 18],
        "days_to_pay": [43, 27, 34, 30, 27],
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)