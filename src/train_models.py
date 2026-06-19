"""
train_models.py

Functions for training and evaluating machine learning models.
"""

import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================================
# Model Training
# ============================================================================

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    """

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    return model, training_time


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    return model, training_time


def train_gradient_boosting(X_train, y_train):
    """
    Train a Gradient Boosting classifier.
    """

    model = GradientBoostingClassifier(
        random_state=42,
    )

    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    return model, training_time


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate a trained classification model.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
        ),
        "auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ),
    }


# ============================================================================
# Training Pipeline
# ============================================================================

def train_all_models(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train and evaluate all classification models.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training features.

    X_test : numpy.ndarray
        Testing features.

    y_train : numpy.ndarray
        Training labels.

    y_test : numpy.ndarray
        Testing labels.

    Returns
    -------
    list
        List containing the evaluation metrics for each model.
    """

    models = {
        "Logistic Regression": train_logistic_regression,
        "Random Forest": train_random_forest,
        "Gradient Boosting": train_gradient_boosting,
    }

    results = []

    for model_name, trainer in models.items():

        model, training_time = trainer(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        results.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
                "auc": metrics["auc"],
                "training_time": training_time,
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )

    return results