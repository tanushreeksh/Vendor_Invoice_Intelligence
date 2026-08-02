import joblib
from pathlib import Path


from data_preprocessing import (
    load_vendor_invoice_data,
    engineer_features,
    prepare_features,
    split_data,
)

from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model,
    cross_validate_model,
)


def main():
    db_path = "data/inventory.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok = True)

    # Load Data
    df = load_vendor_invoice_data(db_path)

    # Feature Engineering
    df = engineer_features(df)

    # Prepare Features
    X, y = prepare_features(df)

    # Split Data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train Models
    models = {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Decision Tree": train_decision_tree(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
    }

    # Cross-validate (5-fold) to pick the most reliable model
    print("Cross-validation results:")
    cv_scores = {
        name: cross_validate_model(model, X, y, name)
        for name, model in models.items()
    }


    print("\nModel Performance...")
    for name, model in models.items():
        evaluate_model(model, X_test, y_test, name)

    # Select best model by cross-validated R2
    best_model_name = max(cv_scores, key=cv_scores.get)
    best_model = models[best_model_name]

    # Save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest Model (by CV R2): {best_model_name} -> saved to {model_path}")


if __name__ == "__main__":
    main()