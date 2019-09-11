import pandas as pd
import numpy as np


def clean_df(
    df,
    exercise_code_level=False,
    drop_level=True,
    date_to_timestamp=True,
    exercise_code_level_lesson=True,
    drop_learning_object=True,
) -> pd.DataFrame:
    df = df.drop(columns=["Unnamed: 0", "id"])
    if exercise_code_level:
        df = add_exercise_code_level(df)
    if date_to_timestamp:
        df = change_date_to_timestamp(df)
        df = df.drop(columns=["created_at"])
    if exercise_code_level_lesson:
        df = add_exercise_code_level_lesson(df)
        df = df.drop(columns=["lesson_id"])
    if drop_level:
        df = df.drop(columns=["level"])
    if drop_learning_object:
        try:
            df = df.drop(columns=["learning_object"])
        except:
            pass
    df["student_id"] = np.unique(df["student_id"], return_inverse=True)[1]
    df = df[df["correctness"].isin((True, False))]
    df["correctness"] = df["correctness"].astype(int)
    return df


def add_exercise_code_level(df) -> pd.DataFrame:
    dataset = df.copy()
    dataset["exercise_code_level"] = (
        dataset["exercise_code"].map(str) + "_" + dataset["level"].map(str)
    )
    return dataset


def add_exercise_code_level_lesson(df) -> pd.DataFrame:
    dataset = df.copy()
    dataset["exercise_code_level_lesson"] = (
        dataset["exercise_code"].map(str)
        + "_"
        + dataset["level"].map(str)
        + "_lesson_"
        + dataset["lesson_id"].map(str)
    )
    return dataset


def change_date_to_timestamp(df) -> pd.DataFrame:
    df["timestamp"] = df["created_at"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
