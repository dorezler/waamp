"""Niestandardowe węzły ComfyUI do przetwarzania danych badań AMP."""

import time
from pathlib import Path

import pandas as pd

from .amp_functions import load_csv_file, save_csv_file


def get_input_csv_paths() -> list[str]:
    """Zwraca listę plików CSV z katalogu input."""
    input_dir = Path("input")
    if not input_dir.exists():
        return [""]
    csv_paths = sorted([str(f.relative_to(".")) for f in input_dir.glob("*.csv")])
    return csv_paths if csv_paths else [""]


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
                "csv_path_from_list": (get_input_csv_paths(), {}),  # Wybór pliku z listy
                "csv_path_from_string": ("STRING", {"default": ""}),  # Ścieżka ręczna (fallback)
            },
        }

    @classmethod
    def IS_CHANGED(cls, **_) -> float:  # pylint: disable=invalid-name
        """Wymusza odświeżenie dropdowna przy każdym uruchomieniu."""
        return time.time()

    def load_csv(self, csv_path_from_list: str, csv_path_from_string: str) -> tuple[pd.DataFrame]:
        """Wczytuje dane z pliku CSV do DataFrame."""
        # Użyj csv_path_from_string jeśli podany, w przeciwnym razie csv_path_from_list
        path = csv_path_from_string.strip() if csv_path_from_string.strip() else csv_path_from_list
        df = load_csv_file(path)
        return (df,)


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


# Mapowanie klas węzłów dla ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoadCSVNode": LoadCSVNode,
    "SaveCSVNode": SaveCSVNode,
}

# Mapowanie nazw wyświetlanych w UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCSVNode": "Load CSV Data",
    "SaveCSVNode": "Save CSV Data",
}
