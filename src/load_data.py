import pandas as pd

ORIGINAL_DATA_DIR = "data\\original_datasets"
ANALYSIS_DATA_DIR = "data\\analysis_datasets"

def load_csv(filename: str) -> pd.DataFrame:

    path = f"{ORIGINAL_DATA_DIR}\\{filename}"
    return pd.read_csv(path)


def load_lap_times() -> pd.DataFrame:
    return load_csv("lap_times.csv")


def load_results() -> pd.DataFrame:
    return load_csv("results.csv")


def load_sprint_results() -> pd.DataFrame:
    return load_csv("sprint_results.csv")


def load_races() -> pd.DataFrame:
    return load_csv("races.csv")


def load_drivers() -> pd.DataFrame:
    return load_csv("drivers.csv")


def load_circuits() -> pd.DataFrame:
    return load_csv("circuits.csv")

