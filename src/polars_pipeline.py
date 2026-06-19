"""
polars_pipeline.py

Complete data processing pipeline implemented with Polars.
"""

import time
import numpy as np
import polars as pl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (
    load_ratings_polars,
    load_movies_polars,
)

from src.feature_engineering import (
    merge_datasets,
    filter_records,
    handle_missing_values,
    create_target_variable,
    compute_user_features,
    compute_movie_features,
    apply_user_features,
    apply_movie_features,
    compute_genre_columns,
    apply_genre_encoding,
    prepare_dataset,
)


# ============================================================================
# Pipeline
# ============================================================================

def run_polars_pipeline(
    ratings_path: str,
    movies_path: str,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Execute the complete Polars preprocessing pipeline.

    Returns
    -------
    tuple
        (
            train_df,
            test_df,
            X_train,
            X_test,
            y_train,
            y_test,
            timings,
        )
    """

    timings = {}

    # ========================================================================
    # Data Loading
    # ========================================================================

    start = time.perf_counter()

    ratings = load_ratings_polars(ratings_path)
    movies = load_movies_polars(movies_path)

    timings["loading"] = time.perf_counter() - start

    # ========================================================================
    # Join
    # ========================================================================

    start = time.perf_counter()

    df = merge_datasets(ratings, movies)

    timings["join"] = time.perf_counter() - start

    # ========================================================================
    # Filtering
    # ========================================================================

    start = time.perf_counter()

    df = filter_records(df)

    timings["filtering"] = time.perf_counter() - start

    # ========================================================================
    # Missing Values
    # ========================================================================

    start = time.perf_counter()

    df = handle_missing_values(df)

    timings["missing_values"] = time.perf_counter() - start

    # ========================================================================
    # Target Variable
    # ========================================================================

    start = time.perf_counter()

    df = create_target_variable(df)

    timings["target"] = time.perf_counter() - start

    # ========================================================================
    # Train/Test Split
    # ========================================================================

    indices = np.arange(df.height)

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=df["liked_movie"].to_numpy(),
    )

    train_df = df[train_idx]
    test_df = df[test_idx]

    # ========================================================================
    # Feature Engineering
    # ========================================================================

    start = time.perf_counter()

    user_features = compute_user_features(train_df)
    movie_features = compute_movie_features(train_df)

    train_df = apply_user_features(train_df, user_features)
    test_df = apply_user_features(test_df, user_features)

    train_df = apply_movie_features(train_df, movie_features)
    test_df = apply_movie_features(test_df, movie_features)

    genre_columns = compute_genre_columns(train_df)

    train_df = apply_genre_encoding(
        train_df,
        genre_columns,
    )

    test_df = apply_genre_encoding(
        test_df,
        genre_columns,
    )

    train_df = prepare_dataset(train_df)
    test_df = prepare_dataset(test_df)

    timings["feature_engineering"] = (
        time.perf_counter() - start
    )

    # ========================================================================
    # Prepare Data for Machine Learning
    # ========================================================================

    X_train = train_df.drop("liked_movie").to_numpy()
    y_train = train_df["liked_movie"].to_numpy()

    X_test = test_df.drop("liked_movie").to_numpy()
    y_test = test_df["liked_movie"].to_numpy()

    # ========================================================================
    # Feature Scaling
    # ========================================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ========================================================================
    # Total Time
    # ========================================================================

    timings["total"] = sum(timings.values())

    return (
        train_df,
        test_df,
        X_train,
        X_test,
        y_train,
        y_test,
        timings,
    )