import joblib
import pandas as pd


MODEL_PATH = "models/predict_freight_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model
    """

    model = joblib.load(model_path)
    return model



def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices

    Parameters:
    input_data : dict or pd.DataFrame with 'Quantity' and 'Dollars' columns

    Returns:
    pd.DataFrame with predicted freight cost
    """

    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df = input_df[["Quantity", "Dollars"]]
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df



if __name__ == "__main__":
    # Example inference run -

    sample_data = {
        "Quantity": [157, 3744, 27479, 1817, 475, 36881, 19427, 99841, 213, 4034],
        "Dollars": [4250.83, 25958.49, 178492.84, 24350.69, 3681.78, 257120.33, 289181.69, 982423.13, 5829.75, 29137.82],
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)