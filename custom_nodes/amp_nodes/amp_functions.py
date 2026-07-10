"""Reużywalne funkcje do przetwarzania danych AMP."""

import logging
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modlamp.descriptors import GlobalDescriptor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def _apply_range_strategy(num1: float, num2: float, strategy: str) -> float:
    """Zastosuj strategię do zakresu wartości."""
    strategy_map = {
        "min": min(num1, num2),
        "max": max(num1, num2),
        "mean": (num1 + num2) / 2,
        "median": (num1 + num2) / 2,
    }
    return strategy_map[strategy]


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


def _parse_activity_value(value: str, range_strategy: str) -> float | None:
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
                return _apply_range_strategy(float(parts[0]), float(parts[1]), range_strategy)
        # Obsługa zakresów: x+/-y, x±y -> x
        if any(op in value for op in ["+/-", "+-", "±"]):
            match = re.match(r"^([\d.]+)", value)
            if match:
                return float(match.group(1))
        # Konwersja bezpośrednia
        return float(value)
    except ValueError:
        return None


def aggregate_duplicates(dataframe: pd.DataFrame, key_column: str, strategy: str = "mean") -> pd.DataFrame:
    """Agreguje duplikaty w DataFrame na podstawie kolumny klucza.

    Args:
        dataframe: DataFrame do agregacji (zakłada kolumny numeryczne poza key_column)
        key_column: Nazwa kolumny klucza (np. "Peptide Sequence")
        strategy: Strategia agregacji: "min", "max", "mean", "median", "first"

    Returns:
        DataFrame z zagregowanymi duplikatami
    """
    key_column = key_column.strip()
    # Walidacja: sprawdź czy key_column nie jest pusty
    if not key_column:
        raise ValueError("No key column provided. Please specify a key column for aggregation.")
    # Walidacja: sprawdź czy kolumna istnieje
    if key_column not in dataframe.columns:
        raise ValueError(f"Column '{key_column}' not found in DataFrame. Available columns: {list(dataframe.columns)}.")
    # Walidacja: sprawdź strategię
    valid_strategies = ["min", "max", "mean", "median", "first"]
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}.")
    # Agreguj duplikaty
    aggregated_df = dataframe.groupby(key_column, as_index=False).agg(strategy)
    # Przygotuj info do logów
    logger.info("Aggregated duplicates by column '%s' with strategy '%s'.", key_column, strategy)
    logger.info("Rows before: %d, after: %d.", len(dataframe), len(aggregated_df))
    return aggregated_df


def clean_activity_values(dataframe: pd.DataFrame, column_name: str, range_strategy: str = "mean") -> pd.DataFrame:
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
    cleaned_df[column_name] = cleaned_df[column_name].apply(lambda x: _parse_activity_value(x, range_strategy))
    # Przygotuj info do logów
    logger.info("Cleaned activity values in column '%s' with strategy '%s'.", column_name, range_strategy)
    logger.info("Rows before: %d, after: %d.", len(dataframe), len(cleaned_df))
    return cleaned_df


def convert_to_binary_classification(
    dataframe: pd.DataFrame, activity_column: str, threshold: float, output_column: str = "Label"
) -> pd.DataFrame:
    """Konwertuje wartości aktywności do klasyfikacji binarnej.

    Args:
        dataframe: DataFrame z wartościami aktywności
        activity_column: Nazwa kolumny z wartościami aktywności
        threshold: Próg aktywności (wartości <= próg są aktywne, wartość 1)
        output_column: Nazwa kolumny wyjściowej z etykietami (domyślnie "Label")

    Returns:
        DataFrame z dodaną kolumną klasyfikacji binarnej (1 = aktywny, 0 = nieaktywny)
    """
    activity_column = activity_column.strip()
    output_column = output_column.strip()
    # Walidacja: sprawdź czy activity_column nie jest pusty
    if not activity_column:
        raise ValueError("No activity column provided. Please specify an activity column name.")
    # Walidacja: sprawdź czy kolumna istnieje
    if activity_column not in dataframe.columns:
        raise ValueError(
            f"Column '{activity_column}' not found in DataFrame. Available columns: {list(dataframe.columns)}."
        )
    # Walidacja: sprawdź threshold
    if threshold <= 0:
        raise ValueError(f"Invalid threshold: {threshold}. Must be > 0.")
    # Walidacja: sprawdź czy output_column nie jest pusty
    if not output_column:
        raise ValueError("No output column name provided. Please specify an output column name.")
    # Konwersja do klasyfikacji binarnej: <= threshold -> 1 (aktywny), > threshold -> 0 (nieaktywny)
    classified_df = dataframe.copy()
    classified_df[output_column] = (classified_df[activity_column] <= threshold).astype(int)
    # Przygotuj info do logów
    active_count = (classified_df[output_column] == 1).sum()
    inactive_count = (classified_df[output_column] == 0).sum()
    logger.info("Converted to binary classification with threshold %.2f in column '%s'.", threshold, activity_column)
    logger.info("Active (1): %d, Inactive (0): %d.", active_count, inactive_count)
    return classified_df


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


def round_column_values(dataframe: pd.DataFrame, column_name: str, decimal_places: int = 2) -> pd.DataFrame:
    """Zaokrągla wartości numeryczne w kolumnie do określonej liczby miejsc po przecinku."""
    column_name = column_name.strip()
    # Walidacja: sprawdź czy column_name nie jest pusty
    if not column_name:
        raise ValueError("No column name provided. Please specify a column name to round values.")
    # Walidacja: sprawdź czy kolumna istnieje
    if column_name not in dataframe.columns:
        raise ValueError(
            f"Column '{column_name}' not found in DataFrame. Available columns: {list(dataframe.columns)}."
        )
    # Walidacja: sprawdź decimal_places
    if decimal_places < 0:
        raise ValueError(f"Invalid decimal_places: {decimal_places}. Must be >= 0.")
    # Zaokrąglij wartości
    rounded_df = dataframe.copy()
    rounded_df[column_name] = rounded_df[column_name].apply(lambda x: round(x, decimal_places) if pd.notna(x) else x)
    # Przygotuj info do logów
    logger.info("Rounded values in column '%s' to %d decimal places.", column_name, decimal_places)
    logger.info("Rows: %d.", len(rounded_df))
    return rounded_df


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


def select_columns(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Wybiera tylko określone kolumny z DataFrame.

    Args:
        dataframe: DataFrame do filtrowania
        columns: Lista nazw kolumn do zachowania

    Returns:
        DataFrame zawierający tylko wybrane kolumny
    """
    # Walidacja: sprawdź czy lista nie jest pusta
    if not columns:
        raise ValueError("No columns provided. Please specify at least one column to select.")
    # Walidacja: sprawdź czy wszystkie kolumny istnieją
    missing_columns = [col for col in columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"Columns {missing_columns} not found in DataFrame. Available columns: {list(dataframe.columns)}."
        )
    # Wybierz tylko określone kolumny
    selected_df = dataframe[columns].copy()
    # Przygotuj info do logów
    logger.info("Selected columns: %s.", columns)
    logger.info("Shape before: %s, after: %s.", dataframe.shape, selected_df.shape)
    return selected_df


def train_random_forest_classifier(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
    vectorized_data: np.ndarray,
    output_dir: str = "output/models",
    model_name: str = "rf_classifier",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int | None = None,
) -> dict:
    """Trenuje klasyfikator Random Forest na zwektoryzowanych danych peptydów.

    Pipeline:
    1. Podział train-test (80:20)
    2. sklearn.Pipeline: StandardScaler → Random Forest
    3. Ewaluacja na zbiorze testowym
    4. Zapis: pipeline.pkl, report.txt, confusion_matrix.png

    Args:
        vectorized_data: Numpy array 2D [features..., label] z vectorize_sequences()
        output_dir: Katalog wyjściowy dla modelu i raportów
        model_name: Nazwa bazowa dla plików (bez rozszerzenia)
        test_size: Proporcja zbioru testowego (0.0-1.0)
        random_state: Seed dla powtarzalności
        n_estimators: Liczba drzew w Random Forest
        max_depth: Maksymalna głębokość drzew (None = bez limitu)

    Returns:
        Dict z metrykami: accuracy, precision, recall, f1, confusion_matrix
    """
    # Walidacja: sprawdź czy vectorized_data nie jest pusty
    if vectorized_data.size == 0:
        raise ValueError("Cannot train on empty data. Please provide vectorized data.")
    # Walidacja: sprawdź wymiary
    if len(vectorized_data.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {vectorized_data.shape}.")
    # Walidacja: sprawdź test_size
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"Invalid test_size: {test_size}. Must be between 0.0 and 1.0.")

    # Utwórz katalog wyjściowy jeśli nie istnieje
    os.makedirs(output_dir, exist_ok=True)

    # Ustaw backend matplotlib na 'Agg' dla pracy w wątku roboczym
    plt.switch_backend("Agg")

    # Rozdziel cechy (X) i etykiety (y)
    X = vectorized_data[:, :-1]  # pylint: disable=invalid-name  # Wszystkie kolumny oprócz ostatniej
    y = vectorized_data[:, -1].astype(int)  # Ostatnia kolumna jako int

    # Podział train-test
    X_train, X_test, y_train, y_test = train_test_split(  # pylint: disable=invalid-name
        X, y, test_size=test_size, random_state=random_state
    )

    # Stwórz Pipeline: StandardScaler → RandomForest
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
                ),
            ),
        ]
    )

    # Trenuj pipeline
    pipeline.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = pipeline.predict(X_test)

    # Oblicz metryki
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)

    # Przygotuj ścieżki do plików
    pipeline_path = os.path.join(output_dir, f"{model_name}_pipeline.pkl")
    report_path = os.path.join(output_dir, f"{model_name}_report.txt")
    confusion_matrix_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")

    # Zapisz pipeline (scaler + model w jednym obiekcie)
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    # Zapisz raport z metrykami
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RANDOM FOREST CLASSIFIER - EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write("Random Forest Parameters:\n")
        f.write(f"  - n_estimators: {n_estimators}\n")
        f.write(f"  - max_depth: {max_depth}\n")
        f.write(f"  - random_state: {random_state}\n\n")
        f.write("Dataset Split:\n")
        f.write(f"  - Train size: {len(X_train)} samples\n")
        f.write(f"  - Test size: {len(X_test)} samples\n")
        f.write(f"  - Test ratio: {test_size:.1%}\n\n")
        f.write("Feature Information:\n")
        f.write(f"  - Number of features: {X.shape[1]}\n")
        f.write("  - Pipeline: StandardScaler → RandomForestClassifier\n\n")
        f.write("=" * 80 + "\n")
        f.write("METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n\n")
        f.write("=" * 80 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(class_report)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"True Negatives (TN):  {conf_matrix[0, 0]}\n")
        f.write(f"False Positives (FP): {conf_matrix[0, 1]}\n")
        f.write(f"False Negatives (FN): {conf_matrix[1, 0]}\n")
        f.write(f"True Positives (TP):  {conf_matrix[1, 1]}\n")

    # Wygeneruj wykres macierzy pomyłek
    fig, ax = plt.subplots(figsize=(8, 6))

    # Użyj ConfusionMatrixDisplay dla ładnego wykresu
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Inactive (0)", "Active (1)"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    # Dodaj tytuł i metryki
    ax.set_title(f"Confusion Matrix - {model_name}\nAccuracy: {accuracy:.4f} | F1: {f1:.4f}", fontsize=12, pad=20)

    # Zapisz wykres
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Przygotuj info do logów
    logger.info("Trained Random Forest classifier with %d estimators", n_estimators)
    logger.info("Train size: %d, Test size: %d (%.1f%%)", len(X_train), len(X_test), test_size * 100)
    logger.info(
        "Test Metrics - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f", accuracy, precision, recall, f1
    )
    logger.info("Saved pipeline: %s", pipeline_path)
    logger.info("Saved report: %s", report_path)
    logger.info("Saved confusion matrix plot: %s", confusion_matrix_path)

    # Zwróć metryki
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix.tolist(),
        "pipeline_path": pipeline_path,
        "report_path": report_path,
        "confusion_matrix_path": confusion_matrix_path,
    }


def vectorize_sequences_esm2(  # pylint: disable=too-many-locals
    dataframe: pd.DataFrame, sequence_column: str, label_column: str
) -> np.ndarray:
    """Wektoryzuje sekwencje peptydów używając ESM2 protein language model.

    Używa ESM2-t6-8M (najmniejszy model, 8M parametrów) do generowania embeddings.
    Każda sekwencja jest reprezentowana jako wektor 320-wymiarowy (mean pooling).

    Args:
        dataframe: DataFrame z sekwencjami i labelkami
        sequence_column: Nazwa kolumny z sekwencjami peptydów
        label_column: Nazwa kolumny z labelkami (0/1)

    Returns:
        Numpy array 2D: [320 features (ESM2 embeddings)..., label]
    """
    sequence_column = sequence_column.strip()
    label_column = label_column.strip()

    # Walidacja: sprawdź czy kolumny nie są puste
    if not sequence_column or not label_column:
        raise ValueError("Both sequence_column and label_column must be provided.")
    # Walidacja: sprawdź czy kolumny istnieją
    for col_name in [sequence_column, label_column]:
        if col_name not in dataframe.columns:
            raise ValueError(
                f"Column '{col_name}' not found in DataFrame. Available columns: {list(dataframe.columns)}."
            )

    # Dynamiczny import - tylko gdy funkcja jest wywoływana
    try:
        import torch  # pylint: disable=import-outside-toplevel
        from transformers import AutoModel, AutoTokenizer  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            "ESM2 requires 'transformers' and 'torch'. Install with: pip install transformers torch"
        ) from e

    # Pobierz sekwencje i labelki
    sequences = dataframe[sequence_column].tolist()
    labels = dataframe[label_column].to_numpy().reshape(-1, 1)

    # Załaduj model ESM2-t6-8M (najmniejszy, najszybszy)
    model_name = "facebook/esm2_t6_8M_UR50D"
    logger.info("Loading ESM2 model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # Tryb ewaluacji (bez treningu)

    # Ustaw urządzenie (CPU lub GPU jeśli dostępne)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Using device: %s", device)

    # Generuj embeddings dla każdej sekwencji
    embeddings_list = []
    with torch.no_grad():  # Wyłącz gradient (oszczędność pamięci)
        for seq in sequences:
            seq = str(seq).strip().upper()
            # Tokenizuj sekwencję
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Przepuść przez model
            outputs = model(**inputs)

            # Pobierz embeddings z ostatniej warstwy (mean pooling po długości sekwencji)
            # outputs.last_hidden_state shape: (batch_size=1, seq_len, hidden_size=320)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling: (1, 320)
            embeddings_list.append(embeddings.cpu().numpy().squeeze())  # Konwertuj do numpy

    # Konwertuj listę do numpy array
    features = np.array(embeddings_list, dtype=np.float32)

    # Połącz cechy z labelkami: [320 features..., label]
    vectorized_data = np.hstack([features, labels])

    # Przygotuj info do logów
    logger.info(
        "Vectorized %d sequences using ESM2-t6-8M (320-dim embeddings)",
        len(sequences),
    )
    logger.info("Feature shape: %s, Final shape: %s", features.shape, vectorized_data.shape)
    logger.info("First 3 rows of vectorized data (first 5 dims):\n%s", vectorized_data[:3, :5])
    logger.info(
        "Data statistics - min: %.3f, max: %.3f, mean: %.3f",
        vectorized_data.min(),
        vectorized_data.max(),
        vectorized_data.mean(),
    )
    return vectorized_data


def vectorize_sequences_aa_composition(  # pylint: disable=too-many-locals
    dataframe: pd.DataFrame, sequence_column: str, label_column: str
) -> np.ndarray:
    """Wektoryzuje sekwencje peptydów do wektora składu aminokwasowego (amino acid composition).

    Oblicza procentowy udział każdego z 20 standardowych aminokwasów w sekwencji.
    Wektor ma 20 cech (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).

    Args:
        dataframe: DataFrame z sekwencjami i labelkami
        sequence_column: Nazwa kolumny z sekwencjami peptydów
        label_column: Nazwa kolumny z labelkami (0/1)

    Returns:
        Numpy array 2D: [20 features (AA composition)..., label] - każdy wiersz to 20 cech + labelka
    """
    sequence_column = sequence_column.strip()
    label_column = label_column.strip()

    # Walidacja: sprawdź czy kolumny nie są puste
    if not sequence_column or not label_column:
        raise ValueError("Both sequence_column and label_column must be provided.")
    # Walidacja: sprawdź czy kolumny istnieją
    for col_name in [sequence_column, label_column]:
        if col_name not in dataframe.columns:
            raise ValueError(
                f"Column '{col_name}' not found in DataFrame. Available columns: {list(dataframe.columns)}."
            )

    # Pobierz sekwencje i labelki
    sequences = dataframe[sequence_column].tolist()
    labels = dataframe[label_column].to_numpy().reshape(-1, 1)

    # Lista standardowych aminokwasów w porządku alfabetycznym
    aa_order = sorted(STANDARD_AMINO_ACIDS)

    # Oblicz skład aminokwasowy dla każdej sekwencji
    composition_matrix = []
    for seq in sequences:
        seq = str(seq).strip().upper()
        seq_len = len(seq)
        if seq_len == 0:
            # Dla pustej sekwencji - same zera
            composition = [0.0] * 20
        else:
            # Zlicz każdy aminokwas i oblicz procentowy udział
            composition = []
            for aa in aa_order:
                count = seq.count(aa)
                percentage = (count / seq_len) * 100.0
                composition.append(percentage)
        composition_matrix.append(composition)

    # Konwertuj do numpy array
    features = np.array(composition_matrix, dtype=np.float64)

    # Połącz cechy z labelkami: [20 features..., label]
    vectorized_data = np.hstack([features, labels])

    # Przygotuj info do logów
    logger.info(
        "Vectorized %d sequences using Amino Acid Composition (20 features: %s)",
        len(sequences),
        ", ".join(aa_order),
    )
    logger.info("Feature shape: %s, Final shape: %s", features.shape, vectorized_data.shape)
    logger.info("First 3 rows of vectorized data:\n%s", vectorized_data[:3])
    logger.info(
        "Data statistics - min: %.3f, max: %.3f, mean: %.3f",
        vectorized_data.min(),
        vectorized_data.max(),
        vectorized_data.mean(),
    )
    return vectorized_data


def vectorize_sequences(dataframe: pd.DataFrame, sequence_column: str, label_column: str) -> np.ndarray:
    """Wektoryzuje sekwencje peptydów do cech numerycznych używając modlamp GlobalDescriptor.

    Oblicza 10 globalnych deskryptorów dla każdej sekwencji:
    - Length, MW, Charge, ChargeDensity, pI, InstabilityInd, Aromaticity, AliphaticInd, BomanInd, HydrophRatio

    Args:
        dataframe: DataFrame z sekwencjami i labelkami
        sequence_column: Nazwa kolumny z sekwencjami peptydów
        label_column: Nazwa kolumny z labelkami (0/1)

    Returns:
        Numpy array 2D: [features..., label] - każdy wiersz to 10 cech + labelka
    """
    sequence_column = sequence_column.strip()
    label_column = label_column.strip()

    # Walidacja: sprawdź czy kolumny nie są puste
    if not sequence_column or not label_column:
        raise ValueError("Both sequence_column and label_column must be provided.")
    # Walidacja: sprawdź czy kolumny istnieją
    for col_name in [sequence_column, label_column]:
        if col_name not in dataframe.columns:
            raise ValueError(
                f"Column '{col_name}' not found in DataFrame. Available columns: {list(dataframe.columns)}."
            )

    # Pobierz sekwencje i labelki
    sequences = dataframe[sequence_column].tolist()
    labels = dataframe[label_column].to_numpy().reshape(-1, 1)

    # Wektoryzuj sekwencje używając modlamp GlobalDescriptor
    descriptor = GlobalDescriptor(sequences)
    descriptor.calculate_all()  # Oblicz wszystkie globalne deskryptory

    # Pobierz macierz cech (n_samples, n_features) i nazwy cech
    features = descriptor.descriptor
    feature_names = descriptor.featurenames

    # Połącz cechy z labelkami: [features..., label]
    vectorized_data = np.hstack([features, labels])

    # Przygotuj info do logów
    logger.info(
        "Vectorized %d sequences using GlobalDescriptor.calculate_all(). Features: %s",
        len(sequences),
        feature_names,
    )
    logger.info("Feature shape: %s, Final shape: %s", features.shape, vectorized_data.shape)
    logger.info("First 3 rows of vectorized data:\n%s", vectorized_data[:3])
    logger.info(
        "Data statistics - min: %.3f, max: %.3f, mean: %.3f",
        vectorized_data.min(),
        vectorized_data.max(),
        vectorized_data.mean(),
    )
    return vectorized_data
