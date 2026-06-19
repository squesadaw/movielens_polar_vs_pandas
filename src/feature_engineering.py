"""
feature_engineering.py

Functions for data integration and feature engineering using Polars.
"""

import polars as pl


# ============================================================================
# Data Integration
# ============================================================================

def merge_datasets(
    ratings: pl.DataFrame,
    movies: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge ratings and movies datasets.
    """

    return ratings.join(
        movies,
        on="movieId",
        how="left",
    )


# ============================================================================
# Data Cleaning
# ============================================================================

def filter_records(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Filter invalid records.
    """

    return df.filter(
        pl.col("rating").is_between(0.5, 5.0)
    )


def handle_missing_values(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Remove missing values.
    """

    return df.drop_nulls()


# ============================================================================
# Target Variable
# ============================================================================

def create_target_variable(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Create binary target variable.

    liked_movie = 1 if rating >= 4.0
    """

    return df.with_columns(
        (
            pl.col("rating") >= 4.0
        )
        .cast(pl.Int8)
        .alias("liked_movie")
    )


# ============================================================================
# User Features
# ============================================================================

def compute_user_features(
    train_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Compute user statistics using ONLY the training set.
    """

    return (
        train_df
        .group_by("userId")
        .agg([
            pl.col("rating")
            .mean()
            .alias("user_avg_rating"),

            pl.len()
            .alias("user_num_ratings"),
        ])
    )


def apply_user_features(
    df: pl.DataFrame,
    user_features: pl.DataFrame,
) -> pl.DataFrame:
    """
    Join user statistics to a dataset.
    """

    return (
        df.join(
            user_features,
            on="userId",
            how="left",
        )
        .with_columns([
            pl.col("user_avg_rating")
            .fill_null(0),

            pl.col("user_num_ratings")
            .fill_null(0)
            .cast(pl.Int32),
        ])
    )


# ============================================================================
# Movie Features
# ============================================================================

def compute_movie_features(
    train_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Compute movie statistics using ONLY the training set.
    """

    return (
        train_df
        .group_by("movieId")
        .agg([
            pl.col("rating")
            .mean()
            .alias("movie_avg_rating"),

            pl.len()
            .alias("movie_num_ratings"),
        ])
    )


def apply_movie_features(
    df: pl.DataFrame,
    movie_features: pl.DataFrame,
) -> pl.DataFrame:
    """
    Join movie statistics to a dataset.
    """

    return (
        df.join(
            movie_features,
            on="movieId",
            how="left",
        )
        .with_columns([
            pl.col("movie_avg_rating")
            .fill_null(0),

            pl.col("movie_num_ratings")
            .fill_null(0)
            .cast(pl.Int32),
        ])
    )


# ============================================================================
# Genre Features
# ============================================================================

def compute_genre_columns(
    train_df: pl.DataFrame,
) -> list[str]:
    """
    Determine the list of genres using ONLY the training set.
    """

    return (
        train_df
        .select("genres")
        .to_series()
        .str.split("|")
        .explode()
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )


def apply_genre_encoding(
    df: pl.DataFrame,
    genres: list[str],
) -> pl.DataFrame:
    """
    Apply one-hot encoding using the genres learned from the training set.
    """

    return df.with_columns([
        pl.col("genres")
        .str.contains(
            rf"(^|\|){genre}(\||$)"
        )
        .cast(pl.Int8)
        .alias(
            "genre_"
            + genre.lower()
                   .replace("-", "_")
                   .replace(" ", "_")
        )

        for genre in genres
    ])


# ============================================================================
# Dataset Preparation
# ============================================================================

def prepare_dataset(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Remove columns not used for machine learning.
    """

    return df.drop([
        "title",
        "genres",
        "timestamp",
        "rating"
    ])