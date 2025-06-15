from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from trainer import evaluate_model
from config import n_jobs


def run_isolation_forest(X_train_dict, X_test_dict, y_train, y_test, feature_key="Statistical"):
    """Train and evaluate an Isolation Forest baseline."""
    X_train = X_train_dict[feature_key]
    X_test = X_test_dict[feature_key]

    # Train only on benign traffic
    X_train_benign = X_train[y_train == 0]

    model = IsolationForest(contamination=0.3, random_state=42, n_jobs=n_jobs)
    model.fit(X_train_benign)

    raw_preds = model.predict(X_test)
    y_pred = (raw_preds == -1).astype(int)

    # Evaluate using overridden predictions (no probability output)
    _, _, metrics = evaluate_model(
        "IsolationForest_Baseline",
        model,
        X_test,
        y_test
    )
    return metrics


def run_one_class_svm(X_train_dict, X_test_dict, y_train, y_test, feature_key="Statistical"):
    """Train and evaluate a One-Class SVM baseline."""
    X_train = X_train_dict[feature_key]
    X_test = X_test_dict[feature_key]

    # Train only on benign traffic
    X_train_benign = X_train[y_train == 0]

    model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.1)
    model.fit(X_train_benign)

    raw_preds = model.predict(X_test)
    y_pred = (raw_preds == -1).astype(int)

    _, _, metrics = evaluate_model(
        "OneClassSVM_Baseline",
        model,
        X_test,
        y_test
    )
    return metrics