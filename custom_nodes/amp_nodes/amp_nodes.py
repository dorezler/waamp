"""Niestandardowe węzły ComfyUI do przetwarzania danych badań AMP."""

import json
import time
from pathlib import Path

import pandas as pd

from .amp_functions import (
    aggregate_duplicates,
    clean_activity_values,
    convert_to_binary_classification,
    convert_units,
    drop_na_in_column,
    filter_column_by_regex,
    filter_standard_amino_acids,
    load_csv_file,
    round_column_values,
    save_csv_file,
    select_columns,
)


class AggregateDuplicatesNode:
    """Węzeł do agregacji duplikatów na podstawie kolumny klucza."""

    # Stałe klasowe definiujące interfejs węzła
    RETURN_TYPES = ("DATAFRAME",)  # Zwraca DataFrame
    RETURN_NAMES = ("dataframe",)  # Nazwa outputu
    FUNCTION = "aggregate_duplicates"  # Nazwa funkcji do wywołania
    CATEGORY = "AMP Research/Data Processing"  # Kategoria w menu ComfyUI

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # pylint: disable=invalid-name
        """Definiuje typy wejściowe dla węzła."""
        return {
            "required": {
                "dataframe": ("DATAFRAME",),  # DataFrame do agregacji
                "key_column": ("STRING", {}),  # Kolumna klucza
                "strategy": (["mean", "min", "max", "median", "first"], {"default": "mean"}),  # Strategia agregacji
            },
        }

    def aggregate_duplicates(self, dataframe: pd.DataFrame, key_column: str, strategy: str) -> tuple[pd.DataFrame]:
        """Agreguje duplikaty w DataFrame na podstawie kolumny klucza."""
        aggregated_df = aggregate_duplicates(dataframe, key_column, strategy)
        return (aggregated_df,)


class CleanActivityValuesNode:
    """Węzeł do czyszczenia wartości Activity - obsługuje zakresy, nierówności."""

    # Stałe klasowe definiujące interfejs węzła
    RETURN_TYPES = ("DATAFRAME",)  # Zwraca DataFrame
    RETURN_NAMES = ("dataframe",)  # Nazwa outputu
    FUNCTION = "clean_activity"  # Nazwa funkcji do wywołania
    CATEGORY = "AMP Research/Data Processing"  # Kategoria w menu ComfyUI

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # pylint: disable=invalid-name
        """Definiuje typy wejściowe dla węzła."""
        return {
            "required": {
                "dataframe": ("DATAFRAME",),  # DataFrame do czyszczenia
                "column_name": ("STRING", {}),  # Nazwa kolumny
                "range_strategy": (["mean", "min", "max", "median"], {"default": "mean"}),  # Strategia dla zakresów
            },
        }

    def clean_activity(self, dataframe: pd.DataFrame, column_name: str, range_strategy: str) -> tuple[pd.DataFrame]:
        """Czyści kolumnę Activity - obsługuje zakresy, nierówności, konwertuje do float."""
        cleaned_df = clean_activity_values(dataframe, column_name, range_strategy)
        return (cleaned_df,)


class ConvertToBinaryClassificationNode:
    """Węzeł do konwersji wartości aktywności do klasyfikacji binarnej."""

    # Stałe klasowe definiujące interfejs węzła
    RETURN_TYPES = ("DATAFRAME",)  # Zwraca DataFrame
    RETURN_NAMES = ("dataframe",)  # Nazwa outputu
    FUNCTION = "convert_to_binary"  # Nazwa funkcji do wywołania
    CATEGORY = "AMP Research/Data Processing"  # Kategoria w menu ComfyUI

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # pylint: disable=invalid-name
        """Definiuje typy wejściowe dla węzła."""
        return {
            "required": {
                "dataframe": ("DATAFRAME",),  # DataFrame do konwersji
                "activity_column": ("STRING", {}),  # Kolumna z aktywnością
                "threshold": ("FLOAT", {}),  # Próg klasyfikacji
                "output_column": ("STRING", {}),  # Nazwa kolumny wyjściowej
            },
        }

    def convert_to_binary(
        self, dataframe: pd.DataFrame, activity_column: str, threshold: float, output_column: str
    ) -> tuple[pd.DataFrame]:
        """Konwertuje wartości aktywności do klasyfikacji binarnej."""
        classified_df = convert_to_binary_classification(dataframe, activity_column, threshold, output_column)
        return (classified_df,)


class ConvertUnitsNode:
    """Węzeł do konwersji jednostek aktywności µg/ml ↔ µM."""

    # Stałe klasowe definiujące interfejs węzła
    RETURN_TYPES = ("DATAFRAME",)  # Zwraca DataFrame
    RETURN_NAMES = ("dataframe",)  # Nazwa outputu
    FUNCTION = "convert_units"  # Nazwa funkcji do wywołania
    CATEGORY = "AMP Research/Data Processing"  # Kategoria w menu ComfyUI

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # pylint: disable=invalid-name
        """Definiuje typy wejściowe dla węzła."""
        return {
            "required": {
                "dataframe": ("DATAFRAME",),  # DataFrame do konwersji
                "sequence_column": ("STRING", {"default": "Peptide Sequence"}),  # Kolumna z sekwencją
                "unit_column": ("STRING", {"default": "Unit"}),  # Kolumna z jednostkami
                "activity_column": ("STRING", {"default": "Activity"}),  # Kolumna z aktywnością
                "target_unit": (["µM", "µg/ml"], {"default": "µM"}),  # Docelowa jednostka
            },
        }

    def convert_units(
        self,
        dataframe: pd.DataFrame,
        sequence_column: str,
        unit_column: str,
        activity_column: str,
        target_unit: str,
    ) -> tuple[pd.DataFrame]:
        """Konwertuje jednostki aktywności używając masy cząsteczkowej peptydu."""
        converted_df = convert_units(dataframe, sequence_column, unit_column, activity_column, target_unit)
        return (converted_df,)


class DropNANode:
    """Węzeł do usuwania wierszy z pustymi wartościami w kolumnie."""

    # Stałe klasowe definiujące interfejs węzła
    RETURN_TYPES = ("DATAFRAME",)  # Zwraca DataFrame
    RETURN_NAMES = ("dataframe",)  # Nazwa outputu
    FUNCTION = "drop_na"  # Nazwa funkcji do wywołania
    CATEGORY = "AMP Research/Data Processing"  # Kategoria w menu ComfyUI

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # pylint: disable=invalid-name
        """Definiuje typy wejściowe dla węzła."""
        return {
            "required": {
                "dataframe": ("DATAFRAME",),  # DataFrame do czyszczenia
                "column_name": ("STRING", {}),  # Nazwa kolumny
            },
        }

    def drop_na(self, dataframe: pd.DataFrame, column_name: str) -> tuple[pd.DataFrame]:
        """Usuwa wiersze z pustymi wartościami w wybranej kolumnie."""
        cleaned_df = drop_na_in_column(dataframe, column_name)
        return (cleaned_df,)


class FilterByRegexNode:
    """Węzeł do filtrowania DataFrame po kolumnie z użyciem regex."""

    # Stałe klasowe definiujące interfejs węzła
    RETURN_TYPES = ("DATAFRAME",)  # Zwraca DataFrame
    RETURN_NAMES = ("dataframe",)  # Nazwa outputu
    FUNCTION = "filter_by_regex"  # Nazwa funkcji do wywołania
    CATEGORY = "AMP Research/Data Processing"  # Kategoria w menu ComfyUI

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # pylint: disable=invalid-name
        """Definiuje typy wejściowe dla węzła."""
        return {
            "required": {
                "dataframe": ("DATAFRAME",),  # DataFrame do filtrowania
                "column_name": ("STRING", {}),  # Nazwa kolumny
                "pattern": ("STRING", {}),  # Wzorzec regex
            },
        }

    def filter_by_regex(self, dataframe: pd.DataFrame, column_name: str, pattern: str) -> tuple[pd.DataFrame]:
        """Filtruje wiersze DataFrame po zawartości kolumny z użyciem wyrażenia regularnego."""
        filtered_df = filter_column_by_regex(dataframe, column_name, pattern)
        return (filtered_df,)


class FilterStandardAminoAcidsNode:
    """Węzeł do filtrowania standardowych aminokwasów."""

    # Stałe klasowe definiujące interfejs węzła
    RETURN_TYPES = ("DATAFRAME",)  # Zwraca DataFrame
    RETURN_NAMES = ("dataframe",)  # Nazwa outputu
    FUNCTION = "filter_standard_amino_acids"  # Nazwa funkcji do wywołania
    CATEGORY = "AMP Research/Data Processing"  # Kategoria w menu ComfyUI

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # pylint: disable=invalid-name
        """Definiuje typy wejściowe dla węzła."""
        return {
            "required": {
                "dataframe": ("DATAFRAME",),  # DataFrame do filtrowania
                "column_name": ("STRING", {}),  # Nazwa kolumny
            },
        }

    def filter_standard_amino_acids(self, dataframe: pd.DataFrame, column_name: str) -> tuple[pd.DataFrame]:
        """Filtruje wiersze zawierające tylko standardowe aminokwasy."""
        filtered_df = filter_standard_amino_acids(dataframe, column_name)
        return (filtered_df,)


class LoadCSVNode:
    """Węzeł do ładowania danych CSV do pandas DataFrame."""

    # Stałe klasowe definiujące interfejs węzła
    RETURN_TYPES = ("DATAFRAME",)  # Zwraca DataFrame
    RETURN_NAMES = ("dataframe",)  # Nazwa outputu
    FUNCTION = "load_csv"  # Nazwa funkcji do wywołania
    CATEGORY = "AMP Research/Data Processing"  # Kategoria w menu ComfyUI

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # pylint: disable=invalid-name
        """Definiuje typy wejściowe dla węzła."""
        return {
            "required": {
                "csv_path_from_list": (cls.get_input_csv_paths(), {}),  # Wybór pliku z listy
                "csv_path_from_string": ("STRING", {"default": ""}),  # Ścieżka ręczna (fallback)
            },
        }

    @classmethod
    def IS_CHANGED(cls, **_) -> float:  # pylint: disable=invalid-name
        """Wymusza odświeżenie dropdowna przy każdym uruchomieniu."""
        return time.time()

    @staticmethod
    def get_input_csv_paths() -> list[str]:
        """Zwraca listę plików CSV z katalogu input."""
        input_dir = Path("input")
        if not input_dir.exists():
            return [""]
        csv_paths = sorted([str(f.relative_to(".")) for f in input_dir.glob("*.csv")])
        return csv_paths if csv_paths else [""]

    def load_csv(self, csv_path_from_list: str, csv_path_from_string: str) -> tuple[pd.DataFrame]:
        """Wczytuje dane z pliku CSV do DataFrame."""
        # Użyj csv_path_from_string jeśli podany, w przeciwnym razie csv_path_from_list
        path = csv_path_from_string.strip() if csv_path_from_string.strip() else csv_path_from_list
        df = load_csv_file(path)
        return (df,)


class RoundValuesNode:
    """Węzeł do zaokrąglania wartości numerycznych w kolumnie."""

    # Stałe klasowe definiujące interfejs węzła
    RETURN_TYPES = ("DATAFRAME",)  # Zwraca DataFrame
    RETURN_NAMES = ("dataframe",)  # Nazwa outputu
    FUNCTION = "round_values"  # Nazwa funkcji do wywołania
    CATEGORY = "AMP Research/Data Processing"  # Kategoria w menu ComfyUI

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # pylint: disable=invalid-name
        """Definiuje typy wejściowe dla węzła."""
        return {
            "required": {
                "dataframe": ("DATAFRAME",),  # DataFrame do zaokrąglenia
                "column_name": ("STRING", {}),  # Nazwa kolumny
                "decimal_places": ("INT", {"default": 2, "min": 0, "max": 10}),  # Miejsca po przecinku
            },
        }

    def round_values(self, dataframe: pd.DataFrame, column_name: str, decimal_places: int) -> tuple[pd.DataFrame]:
        """Zaokrągla wartości numeryczne w kolumnie do określonej liczby miejsc po przecinku."""
        rounded_df = round_column_values(dataframe, column_name, decimal_places)
        return (rounded_df,)


class SaveCSVNode:
    """Węzeł do zapisywania DataFrame do pliku CSV."""

    # Stałe klasowe definiujące interfejs węzła
    RETURN_TYPES = ()  # Brak outputów
    FUNCTION = "save_csv"  # Nazwa funkcji do wywołania
    CATEGORY = "AMP Research/Data Processing"  # Kategoria w menu ComfyUI
    OUTPUT_NODE = True  # Oznacz jako węzeł wyjściowy (końcowy w workflow)

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # pylint: disable=invalid-name
        """Definiuje typy wejściowe dla węzła."""
        return {
            "required": {
                "dataframe": ("DATAFRAME",),  # DataFrame do zapisania
                "output_path": ("STRING", {}),  # Ścieżka wyjściowa
                "include_index": (["False", "True"], {"default": "False"}),  # Czy zapisać indeks
                "overwrite": (["False", "True"], {"default": "False"}),  # Czy nadpisać istniejący plik
            },
        }

    def save_csv(self, dataframe: pd.DataFrame, output_path: str, include_index: str, overwrite: str) -> dict:
        """Zapisuje DataFrame do pliku CSV."""
        include_index_bool = include_index == "True"
        overwrite_bool = overwrite == "True"
        save_csv_file(dataframe, output_path, include_index=include_index_bool, overwrite=overwrite_bool)
        return {}


class SelectColumnsNode:
    """Węzeł do wyboru określonych kolumn z DataFrame."""

    # Stałe klasowe definiujące interfejs węzła
    RETURN_TYPES = ("DATAFRAME",)  # Zwraca DataFrame
    RETURN_NAMES = ("dataframe",)  # Nazwa outputu
    FUNCTION = "select_columns"  # Nazwa funkcji do wywołania
    CATEGORY = "AMP Research/Data Processing"  # Kategoria w menu ComfyUI

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # pylint: disable=invalid-name
        """Definiuje typy wejściowe dla węzła."""
        return {
            "required": {
                "dataframe": ("DATAFRAME",),  # DataFrame do filtrowania
                "columns": ("STRING", {}),  # Lista kolumn w formacie JSON
            },
        }

    def select_columns(self, dataframe: pd.DataFrame, columns: str) -> tuple[pd.DataFrame]:
        """Wybiera tylko określone kolumny z DataFrame."""
        columns_list = json.loads(columns)
        selected_df = select_columns(dataframe, columns_list)
        return (selected_df,)


# Mapowanie klas węzłów dla ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoadCSVNode": LoadCSVNode,
    "FilterByRegexNode": FilterByRegexNode,
    "FilterStandardAminoAcidsNode": FilterStandardAminoAcidsNode,
    "DropNANode": DropNANode,
    "CleanActivityValuesNode": CleanActivityValuesNode,
    "ConvertUnitsNode": ConvertUnitsNode,
    "RoundValuesNode": RoundValuesNode,
    "SelectColumnsNode": SelectColumnsNode,
    "AggregateDuplicatesNode": AggregateDuplicatesNode,
    "ConvertToBinaryClassificationNode": ConvertToBinaryClassificationNode,
    "SaveCSVNode": SaveCSVNode,
}

# Mapowanie nazw wyświetlanych w UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCSVNode": "Load CSV Data",
    "FilterByRegexNode": "Filter by Regex",
    "FilterStandardAminoAcidsNode": "Filter Standard Amino Acids",
    "DropNANode": "Drop NA Values",
    "CleanActivityValuesNode": "Clean Activity Values",
    "ConvertUnitsNode": "Convert Activity Units",
    "RoundValuesNode": "Round Values",
    "SelectColumnsNode": "Select Columns",
    "AggregateDuplicatesNode": "Aggregate Duplicates",
    "ConvertToBinaryClassificationNode": "Convert to Binary Classification",
    "SaveCSVNode": "Save CSV Data",
}
