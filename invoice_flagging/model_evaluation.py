from sklearn.experimental import enable_halving_search_cv  # noqa: required to unlock HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score



def train_random_forest_model(X_train, y_train):
    # n_jobs=1 (default) here on purpose - parallelism happens at the
    # search level below. Setting n_jobs=-1 on both the forest and the
    # search makes them compete for the same cores and can be slower.
    rf = RandomForestClassifier(
        random_state = 42, class_weight = "balanced"
    )

    # Trimmed grid: dropped "entropy" (rarely beats gini, costlier to
    # compute) and thinned min_samples_split/min_samples_leaf to the
    # values that matter most. 3 x 4 x 2 x 2 x 1 = 48 combinations,
    # down from 216.
    param_grid = {
        "n_estimators" : [100, 200, 300],
        "max_depth" : [None, 4, 5, 6],
        "min_samples_split" : [2, 5],
        "min_samples_leaf" : [1, 5],
        "criterion" : ["gini"]
    }

    scorer = make_scorer(f1_score)

    # HalvingGridSearchCV trains all 48 candidates on a small subset of
    # the data first, discards the weak performers, then re-trains the
    # survivors on progressively larger subsets. Same grid, same scoring,
    # same cv - usually 3-5x faster than exhaustive GridSearchCV for an
    # equivalent result.
    grid_search = HalvingGridSearchCV(
        estimator = rf,
        param_grid = param_grid,
        scoring = scorer,
        cv = 5,
        factor = 3,
        verbose = 0,
        n_jobs = -1,
        random_state = 42,
    )

    grid_search.fit(X_train, y_train)
    return grid_search



def evaluate_classifier(model, X_test, y_test, model_name):
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy: .2f}")
    print(report)