from pathlib import Path
import joblib

from data_preprocessing import apply_labels, load_invoice_data, split_data
from model_evaluation import train_random_forest_model, evaluate_classifier



FEATURES = [
    "invoice_quantity",
    "Freight",
    "total_brands",
    "total_item_quantity",
    "days_po_to_invoice",
    "days_to_pay"
]

TARGET = "flag_invoice"



def main():
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Load Data
    df = load_invoice_data()
    df = apply_labels(df)

    # Prepare Data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    # Train and evaluate models
    grid_search = train_random_forest_model(X_train, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test,
        y_test,
        "Random Forest Classifier"
    )

    # Save best model
    joblib.dump(grid_search.best_estimator_, model_dir / "flag_invoice.pkl")



if __name__ == "__main__":
    main()