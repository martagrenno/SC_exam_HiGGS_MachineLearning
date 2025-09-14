"""
Functions of the pipline to be used in the train.py
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_normalize_split(file_path: str, test_size: float, random_seed: int):
    if not os.path.exists("train.csv") or not os.path.exists("test.csv"):
        df = load_data_csv(file_path)
        df = normalize(df)
        df_train, df_test = data_split(df, test_size=test_size, random_seed=random_seed, save=True)
    else:
        print("\033[94mReading the existing 'train.csv' and 'test.csv'...\033[0m")
        df_train = pd.read_csv("train.csv", header=None, dtype=float)
        df_test = pd.read_csv("test.csv", header=None, dtype=float)
        print("\033[92mReading completed!\033[0m")
    
    return (df_train, df_test)


def load_data_csv(file_path: str) -> pd.DataFrame:
    try:
        print("\033[94mReading the input file...\033[0m")
        df = pd.read_csv(file_path, header=None, dtype=float)
        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
            raise ValueError("All columns in the CSV must be numeric.")
        print("\033[92mReading completed!\033[0m")
        
    except Exception as e:
        raise RuntimeError(f"Error loading CSV file: {file_path}\n{e}")
    return df

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    print("\033[94mNormalizing data...\033[0m")
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(df)
    print("\033[92mData normalized!\033[0m")
    return pd.DataFrame(features_scaled)

def data_split(df: pd.DataFrame = None, test_size: float = 0.2 , random_seed: int = None, save: bool=True):
    if  not os.path.exists("train.csv") or not os.path.exists("test.csv"):
        print("\033[94mSplitting data...\033[0m")
        train, test = train_test_split(df, test_size=test_size, random_state=random_seed)
        if save:
            train.to_csv("train.csv", index=False)
            test.to_csv("test.csv", index=False)
        print("\033[92mData splitted!\033[0m")
    else:
        print("\033[94mtrain.csv and test.csv already exist, reading them instead\033[0m")
        train = pd.read_csv("train.csv", header=None, dtype=float)
        test = pd.read_csv("test.csv", header=None, dtype=float)

    return (train, test)
