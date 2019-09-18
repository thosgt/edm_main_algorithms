import pandas as pd
import numpy as np


def clean_df(
    df: pd.DataFrame, drop_level=True, drop_learning_object=True, verbose=True
) -> pd.DataFrame:
    df = df.drop(columns=["Unnamed: 0", "id", "created_at"])
    df = add_exercise_code_level_lesson(df)
    df = df.drop(columns=["lesson_id", "level"])
    if drop_learning_object:
        try:
            df = df.drop(columns=["learning_object"])
        except:
            pass
    df["student_id"] = np.unique(df["student_id"], return_inverse=True)[1]
    df = df[df["correctness"].isin((True, False))]
    df["correctness"] = df["correctness"].astype(int)
    if verbose:
        print(
            f'There are {len(df["student_id"].unique())} students and {len(df)} traces in this dataset'
        )
    return df


def add_exercise_code_level_lesson(df: pd.DataFrame) -> pd.DataFrame:
    dataset = df.copy()
    dataset["exercise_code_level_lesson"] = (
        dataset["exercise_code"].map(str)
        + "_"
        + dataset["level"].map(str)
        + "_lesson_"
        + dataset["lesson_id"].map(str)
    )
    return dataset
