import pandas as pd
import numpy as np


def clean_df(df, exercise_code_level=True, drop_level=True, date_to_timestamp=True):
    df = df.drop(columns=['Unnamed: 0', 'id'])
    if exercise_code_level:
        df = add_exercise_code_level(df)
    if drop_level:
        df = df.drop(columns=['level'])
    if date_to_timestamp:
        df = change_date_to_timestamp(df)
        df = df.drop(columns=['created_at'])
    df["student_id"] = np.unique(df["student_id"], return_inverse=True)[1]
    return df

def add_exercise_code_level(df):
    dataset = df.copy()
    dataset['exercise_code_level'] = dataset['exercise_code'].map(str) + '_' + dataset['level'].map(str)
    return dataset

def change_date_to_timestamp(df):
    df["timestamp"] = df["created_at"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
    return df
