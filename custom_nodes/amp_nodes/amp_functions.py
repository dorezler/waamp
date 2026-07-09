"""Reużywalne funkcje do przetwarzania danych AMP."""

import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Masy monoisotopowe aminokwasów (Da) - pełna masa (monoisotopic)
AMINO_ACID_WEIGHTS = {
    "A": 89.047,  # Alanine
    "C": 121.020,  # Cysteine
    "D": 133.038,  # Aspartic acid
    "E": 147.053,  # Glutamic acid
    "F": 165.079,  # Phenylalanine
    "G": 75.032,  # Glycine
    "H": 155.069,  # Histidine
    "I": 131.095,  # Isoleucine
    "K": 146.106,  # Lysine
    "L": 131.095,  # Leucine
    "M": 149.051,  # Methionine
    "N": 132.053,  # Asparagine
    "P": 115.063,  # Proline
    "Q": 146.069,  # Glutamine
    "R": 174.112,  # Arginine
    "S": 105.043,  # Serine
    "T": 119.058,  # Threonine
    "V": 117.079,  # Valine
    "W": 204.090,  # Tryptophan
    "Y": 181.074,  # Tyrosine
}
# Standardowe aminokwasy (20 naturalnych, tylko wielkie litery)
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _apply_range_strategy(num1: float, num2: float, strategy: str, decimal_places: int) -> float:
    """Zastosuj strategię do zakresu wartości."""
    strategy_map = {
        "min": min(num1, num2),
        "max": max(num1, num2),
        "mean": (num1 + num2) / 2,
        "median": (num1 + num2) / 2,
    }
    return round(strategy_map[strategy], decimal_places)


def _calculate_molecular_weight(sequence: str) -> float:
    """Oblicza masę cząsteczkową peptydu z sekwencji aminokwasów.

    MW = Σ(masy aminokwasów) - (n-1) × 18.015
    gdzie n = liczba aminokwasów, 18.015 = masa H2O (strata przy tworzeniu wiązania peptydowego)
    """
    if pd.isna(sequence) or not sequence:
        return 0.0
    sequence = str(sequence).strip().upper()
    # Suma mas aminokwasów
    total_weight = sum(AMINO_ACID_WEIGHTS.get(aa, 0.0) for aa in sequence)
    # Odejmij masę H2O za każde wiązanie peptydowe (n-1 wiązań dla n aminokwasów)
    num_peptide_bonds = len(sequence) - 1
    total_weight -= num_peptide_bonds * 18.015
    return total_weight


def _convert_unit_for_row(current_unit: str, activity: float, mw: float, target_unit: str) -> float:
    """Konwertuje jednostkę aktywności dla pojedynczego wiersza.

    Args:
        current_unit: Aktualna jednostka z wiersza
        activity: Wartość aktywności
        mw: Masa cząsteczkowa peptydu
        target_unit: Docelowa jednostka

    Returns:
        Skonwertowana wartość aktywności
    """
    # Pomijanie wierszy bez MW lub Activity
    if pd.isna(activity) or mw == 0.0:
        return activity
    # Bez konwersji jeśli jednostki są takie same
    if current_unit == target_unit:
        return activity
    # Konwersja µg/ml → µM
    if current_unit == "µg/ml" and target_unit == "µM":
        return activity * 1000 / mw
    # Konwersja µM → µg/ml
    if current_unit == "µM" and target_unit == "µg/ml":
        return activity * mw / 1000
    # Nieznana jednostka - zwróć oryginalną wartość
    return activity


def _parse_activity_value(value: str, range_strategy: str, decimal_places: int) -> float | None:
    """Parsuje pojedynczą wartość Activity do float."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    # Usuń spacje i normalizuj myślniki
    value = value.replace(" ", "").replace("–", "-").replace("—", "-")
    # Obsługa jawnego "NA"
    if value.upper() == "NA":
        return None
    # Obsługa nierówności: >x, <x, >=x, <=x -> x (usuń operator, zostaw wartość/zakres)
    if value.startswith((">", "<")):
        value = re.sub(r"^[><]=?", "", value)
    try:
        # Obsługa zakresów: x-y lub x->y
        if "-" in value and not value.startswith("-"):
            parts = re.split(r"->|-", value)
            if len(parts) == 2:
                return _apply_range_strategy(float(parts[0]), float(parts[1]), range_strategy, decimal_places)
        # Obsługa zakresów: x+/-y, x±y -> x
        if any(op in value for op in ["+/-", "+-", "±"]):
            match = re.match(r"^([\d.]+)", value)
            if match:
                return round(float(match.group(1)), decimal_places)
        # Konwersja bezpośrednia
        return round(float(value), decimal_places)
    except ValueError:
        return None


def clean_activity_values(
    dataframe: pd.DataFrame, column_name: str, range_strategy: str = "mean", decimal_places: int = 2
) -> pd.DataFrame:
    """Czyści kolumnę Activity - obsługuje zakresy, nierówności, konwertuje do float."""
    column_name = column_name.strip()
    # Walidacja: sprawdź czy column_name nie jest pusty
    if not column_name:
        raise ValueError("No column name provided. Please specify a column name to clean activity values.")
    # Walidacja: sprawdź czy kolumna istnieje
    if column_name not in dataframe.columns:
        raise ValueError(
            f"Column '{column_name}' not found in DataFrame. Available columns: {list(dataframe.columns)}."
        )
    # Walidacja: sprawdź strategię dla zakresów
    valid_strategies = ["min", "max", "mean", "median"]
    if range_strategy not in valid_strategies:
        raise ValueError(f"Invalid range_strategy: {range_strategy}. Must be one of {valid_strategies}.")
    # Zastosuj funkcję czyszczącą
    cleaned_df = dataframe.copy()
    cleaned_df[column_name] = cleaned_df[column_name].apply(
        lambda x: _parse_activity_value(x, range_strategy, decimal_places)
    )
    # Przygotuj info do logów
    logger.info("Cleaned activity values in column '%s' with strategy '%s'.", column_name, range_strategy)
    logger.info("Rows before: %d, after: %d.", len(dataframe), len(cleaned_df))
    return cleaned_df


def convert_units(
    dataframe: pd.DataFrame,
    sequence_column: str,
    unit_column: str,
    activity_column: str,
    target_unit: str,
) -> pd.DataFrame:
    """Konwertuje jednostki aktywności µg/ml ↔ µM używając masy cząsteczkowej peptydu."""
    sequence_column = sequence_column.strip()
    unit_column = unit_column.strip()
    activity_column = activity_column.strip()
    target_unit = target_unit.strip()
    # Walidacja: sprawdź czy kolumny nie są puste
    if not sequence_column or not unit_column or not activity_column or not target_unit:
        raise ValueError("All column names and target unit must be provided.")
    # Walidacja: sprawdź czy kolumny istnieją
    for col_name in [sequence_column, unit_column, activity_column]:
        if col_name not in dataframe.columns:
            raise ValueError(
                f"Column '{col_name}' not found in DataFrame. Available columns: {list(dataframe.columns)}."
            )
    # Walidacja: sprawdź target_unit
    valid_units = ["µM", "µg/ml"]
    if target_unit not in valid_units:
        raise ValueError(f"Invalid target_unit: {target_unit}. Must be one of {valid_units}.")
    converted_df = dataframe.copy()
    # Oblicz masy cząsteczkowe dla wszystkich peptydów
    converted_df["_mw"] = converted_df[sequence_column].apply(_calculate_molecular_weight)
    # Zastosuj konwersję jednostek
    converted_df[activity_column] = converted_df.apply(
        lambda row: _convert_unit_for_row(str(row[unit_column]).strip(), row[activity_column], row["_mw"], target_unit),
        axis=1,
    )
    # Zaktualizuj kolumnę Unit
    converted_df[unit_column] = converted_df[unit_column].apply(
        lambda x: target_unit if str(x).strip() in ["µM", "µg/ml"] else x
    )
    # Usuń tymczasową kolumnę _mw
    converted_df = converted_df.drop(columns=["_mw"])
    # Przygotuj info do logów
    logger.info("Converted units in column '%s' to '%s'.", activity_column, target_unit)
    logger.info("Rows: %d.", len(converted_df))
    return converted_df


def drop_na_in_column(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Usuwa wiersze z pustymi wartościami w wybranej kolumnie."""
    column_name = column_name.strip()
    # Walidacja: sprawdź czy column_name nie jest pusty
    if not column_name:
        raise ValueError("No column name provided. Please specify a column name to drop NA values from.")
    # Walidacja: sprawdź czy kolumna istnieje
    if column_name not in dataframe.columns:
        raise ValueError(
            f"Column '{column_name}' not found in DataFrame. Available columns: {list(dataframe.columns)}."
        )
    # Usuń wiersze z NA w kolumnie
    cleaned_df = dataframe.dropna(subset=column_name).copy()
    # Przygotuj info do logów
    logger.info("Dropped NA values in column '%s'.", column_name)
    logger.info("Rows before: %d, after: %d.", len(dataframe), len(cleaned_df))
    return cleaned_df


def filter_standard_amino_acids(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Usuwa wiersze zawierające niestandardowe aminokwasy (x, X, małe litery)."""
    column_name = column_name.strip()
    # Walidacja: sprawdź czy column_name nie jest pusty
    if not column_name:
        raise ValueError("No column name provided. Please specify a column name to filter standard amino acids.")
    # Walidacja: sprawdź czy kolumna istnieje
    if column_name not in dataframe.columns:
        raise ValueError(
            f"Column '{column_name}' not found in DataFrame. Available columns: {list(dataframe.columns)}."
        )
    # Filtruj tylko standardowe aminokwasy
    pattern = re.compile(f"^[{STANDARD_AMINO_ACIDS}]+$")
    mask = dataframe[column_name].astype(str).str.match(pattern, na=False)
    filtered_df = dataframe[mask].copy()
    # Przygotuj info do logów
    logger.info("Filtered standard amino acids in column '%s'.", column_name)
    logger.info("Rows before: %d, after: %d.", len(dataframe), len(filtered_df))
    return filtered_df


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
