"""
preprocessing.py

Functions for loading datasets and retrieving general dataset/system
information used throughout the project.
"""

from pathlib import Path
import os

import pandas as pd
import polars as pl
import psutil


# ============================================================================
# Data Loading
# ============================================================================

def load_ratings_polars(path: str | Path) -> pl.DataFrame:
    """
    Load the ratings dataset using Polars.

    Parameters
    ----------
    path : str or Path
        Path to ratings.csv.

    Returns
    -------
    pl.DataFrame
    """
    return pl.read_csv(path)


def load_movies_polars(path: str | Path) -> pl.DataFrame:
    """
    Load the movies dataset using Polars.

    Parameters
    ----------
    path : str or Path
        Path to movies.csv.

    Returns
    -------
    pl.DataFrame
    """
    return pl.read_csv(path)


def load_ratings_pandas(path: str | Path) -> pd.DataFrame:
    """
    Load the ratings dataset using Pandas.

    Parameters
    ----------
    path : str or Path
        Path to ratings.csv.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)


def load_movies_pandas(path: str | Path) -> pd.DataFrame:
    """
    Load the movies dataset using Pandas.

    Parameters
    ----------
    path : str or Path
        Path to movies.csv.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)


# ============================================================================
# Dataset Information
# ============================================================================

def get_dataset_info(df) -> dict:
    """
    Return general information about a dataset.

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame

    Returns
    -------
    dict
        Dictionary containing:
        - rows
        - columns
        - column_names
    """

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
    }


def get_missing_values(df):
    """
    Return the number of missing values for each column.

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame

    Returns
    -------
    pandas.Series or polars.DataFrame
    """

    if isinstance(df, pl.DataFrame):
        return df.null_count()

    return df.isnull().sum()


def get_dataset_size(path: str | Path) -> float:
    """
    Return the dataset size in MB.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    float
        Dataset size in MB.
    """

    return Path(path).stat().st_size / (1024 ** 2)


# ============================================================================
# System Information
# ============================================================================

def get_system_info() -> dict:
    """
    Return system information for benchmarking.

    Returns
    -------
    dict
        Dictionary containing:
        - cpu_cores
        - total_ram_gb
    """

    return {
        "cpu_cores": os.cpu_count(),
        "total_ram_gb": round(
            psutil.virtual_memory().total / (1024 ** 3),
            2,
        ),
    }