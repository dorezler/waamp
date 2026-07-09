"""Reużywalne funkcje do przetwarzania danych AMP."""

import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)


def filter_column_by_regex(dataframe: pd.DataFrame, column_name: str, pattern: str) -> pd.DataFrame:
    """Filtruje wiersze DataFrame po zawartości kolumny z użyciem wyrażenia regularnego."""
    pattern = pattern.strip()
    column_name = column_name.strip()
    # Walidacja: sprawdź czy column_name nie jest pusty
    if not column_name:
        raise ValueError("No column name provided. Please specify a column name to filter by.")
    # Walidacja: sprawdź czy pattern nie jest pusty
    if not pattern:
        raise ValueError("No pattern provided. Please specify a regex pattern to filter by.")
    # Walidacja: sprawdź czy kolumna istnieje
    if column_name not in dataframe.columns:
        raise ValueError(
            f"Column '{column_name}' not found in DataFrame. Available columns: {list(dataframe.columns)}."
        )
    # Walidacja: sprawdź czy pattern jest poprawnym wyrażeniem regularnym
    try:
        compiled_pattern = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {pattern}. Error: {e}.") from e
    # Filtruj DataFrame
    mask = dataframe[column_name].astype(str).str.contains(compiled_pattern, na=False, regex=True)
    filtered_df = dataframe[mask].copy()
    # Przygotuj info do logów
    logger.info("Filtered by column '%s' with pattern '%s'.", column_name, pattern)
    logger.info("Rows before: %d, after: %d.", len(dataframe), len(filtered_df))
    return filtered_df


def load_csv_file(csv_path: str) -> pd.DataFrame:
    """Wczytuje dane z pliku CSV do DataFrame."""
    csv_path = csv_path.strip()
    # Walidacja: sprawdź czy ścieżka nie jest pusta
    if not csv_path:
        raise ValueError("No CSV file path provided. Please specify a path to a CSV file.")
    # Walidacja: sprawdź czy plik istnieje
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}.")
    # Wczytaj CSV do DataFrame
    df = pd.read_csv(csv_path)
    # Przygotuj info do logów
    logger.info("Loaded CSV file: %s.", csv_path)
    logger.info("Shape: %s, Columns: %s.", df.shape, list(df.columns))
    return df


def save_csv_file(
    dataframe: pd.DataFrame, output_path: str, include_index: bool = False, overwrite: bool = False
) -> None:
    """Zapisuje DataFrame do pliku CSV."""
    output_path = output_path.strip()
    # Walidacja: sprawdź czy ścieżka nie jest pusta
    if not output_path:
        raise ValueError("No output path provided. Please specify a path to save the CSV file.")
    # Walidacja: sprawdź czy DataFrame nie jest pusty
    if dataframe.empty:
        raise ValueError("Cannot save empty DataFrame. Please provide a DataFrame with data.")
    # Sprawdź czy plik istnieje
    file_exists = os.path.exists(output_path)
    # Walidacja: sprawdź czy plik już nie istnieje (jeśli overwrite=False)
    if not overwrite and file_exists:
        raise FileExistsError(f"Output file already exists: {output_path}. Set overwrite=True or use a different path.")
    # Utwórz katalog jeśli nie istnieje
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Zapisz DataFrame do CSV
    dataframe.to_csv(output_path, index=include_index)
    # Przygotuj info do logów
    if overwrite and file_exists:
        logger.warning("Overwritten CSV file: %s.", output_path)
    else:
        logger.info("Saved CSV file: %s.", output_path)
    logger.info("Shape: %s, Columns: %s.", dataframe.shape, list(dataframe.columns))
