from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model



def train_decision_tree(X_train, y_train, max_depth=5):
    model = DecisionTreeRegressor(
        max_depth=max_depth, random_state=42
        )
    model.fit(X_train, y_train)
    return model



def train_random_forest(X_train, y_train, max_depth=6):
    model = RandomForestRegressor(
        max_depth=max_depth, random_state=42
        )
    model.fit(X_train, y_train)
    return model



def cross_validate_model(model, X, y, model_name: str, cv=5) -> float:
    """
    Return mean 5-fold cross-validated R2. Used to pick the final model
    instead of relying on a single train/test split.
    """

    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
    mean_r2 = float(scores.mean()) * 100

    print(f"{model_name}: {mean_r2:.2f}%")

    return mean_r2



def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate Regression model and return metrics
    """

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = root_mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred) * 100

    print(f"\n{model_name} -")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}%")

    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }