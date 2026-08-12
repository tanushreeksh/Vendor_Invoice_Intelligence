import joblib
import pandas as pd


MODEL_PATH = "models/flag_invoice.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained invoice risk classifier model
    """

    model = joblib.load(model_path)
    return model



def predict_invoice_flag(input_data, lower_threshold: float = 0.30, upper_threshold: float = 0.40):
    """
    Predict risk flag for new vendor invoices

    Parameters:
    input_data      : dict
    lower_threshold : below this probability an invoice is confidently
                       "not risky" (0). 
    upper_threshold : at or above this probability an invoice is
                       confidently "risky" (1). 

    Returns:
    pd.DataFrame with predicted invoice flag (0 = not risky, 1 = risky,
    "review" = falls between the two evaluated thresholds) and the
    underlying probability.
    """

    model = load_model()
    input_df = pd.DataFrame(input_data)
    probs = model.predict_proba(input_df)[:, 1]
    input_df['Risk_Probability'] = probs.round(3)

    flags = pd.Series(probs, index=input_df.index).apply(
        lambda p: 1 if p >= upper_threshold else (0 if p < lower_threshold else "review")
    )
    input_df['Predicted_Invoice_Flag'] = flags
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