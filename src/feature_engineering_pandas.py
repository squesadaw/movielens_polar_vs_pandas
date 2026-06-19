"""
feature_engineering_pandas.py

Functions for data integration and feature engineering using Pandas.
"""

import pandas as pd


# ============================================================================
# Data Integration
# ============================================================================

def merge_datasets(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge ratings and movies datasets.
    """

    return ratings.merge(
        movies,
        on="movieId",
        how="left",
    )


# ============================================================================
# Data Cleaning
# ============================================================================

def filter_records(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter invalid records.
    """

    return df[
        df["rating"].between(0.5, 5.0)
    ].copy()


def handle_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove missing values.
    """

    return df.dropna().copy()


# ============================================================================
# Target Variable
# ============================================================================

def create_target_variable(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create binary target variable.

    liked_movie = 1 if rating >= 4.0
    """

    df = df.copy()

    df["liked_movie"] = (
        df["rating"] >= 4.0
    ).astype("int8")

    return df


# ============================================================================
# User Features
# ============================================================================

def compute_user_features(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute user statistics using ONLY the training set.
    """

    return (
        train_df
        .groupby("userId")
        .agg(
            user_avg_rating=("rating", "mean"),
            user_num_ratings=("rating", "count"),
        )
        .reset_index()
    )


def apply_user_features(
    df: pd.DataFrame,
    user_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join user statistics to a dataset.
    """

    df = df.merge(
        user_features,
        on="userId",
        how="left",
    )

    df["user_avg_rating"] = (
        df["user_avg_rating"]
        .fillna(0)
    )

    df["user_num_ratings"] = (
        df["user_num_ratings"]
        .fillna(0)
        .astype("int32")
    )

    return df


# ============================================================================
# Movie Features
# ============================================================================

def compute_movie_features(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute movie statistics using ONLY the training set.
    """

    return (
        train_df
        .groupby("movieId")
        .agg(
            movie_avg_rating=("rating", "mean"),
            movie_num_ratings=("rating", "count"),
        )
        .reset_index()
    )


def apply_movie_features(
    df: pd.DataFrame,
    movie_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join movie statistics to a dataset.
    """

    df = df.merge(
        movie_features,
        on="movieId",
        how="left",
    )

    df["movie_avg_rating"] = (
        df["movie_avg_rating"]
        .fillna(0)
    )

    df["movie_num_ratings"] = (
        df["movie_num_ratings"]
        .fillna(0)
        .astype("int32")
    )

    return df


# ============================================================================
# Genre Features
# ============================================================================

def compute_genre_columns(
    train_df: pd.DataFrame,
) -> list[str]:
    """
    Determine the list of genres using ONLY the training set.
    """

    return sorted({
        genre
        for values in train_df["genres"].dropna()
        for genre in values.split("|")
    })


def apply_genre_encoding(
    df: pd.DataFrame,
    genres: list[str],
) -> pd.DataFrame:
    """
    Apply one-hot encoding using the genres learned from the training set.
    """

    df = df.copy()

    for genre in genres:

        column_name = (
            "genre_"
            + genre.lower()
                   .replace("-", "_")
                   .replace(" ", "_")
        )

        df[column_name] = (
            df["genres"]
            .str.contains(
                rf"(^|\|){genre}(\||$)",
                regex=True,
            )
            .astype("int8")
        )

    return df


# ============================================================================
# Dataset Preparation
# ============================================================================

def prepare_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove columns not used for machine learning.
    """

    return df.drop(
        columns=[
            "title",
            "genres",
            "timestamp",
            "rating"
        ]
    )