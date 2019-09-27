import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

def clean_df(
    df,
    exercise_code_level=False,
    date_to_timestamp=True,
) -> pd.DataFrame:
    df = df[df["correctness"].isin((True, False))].copy()
    df["correctness"] = df["correctness"].astype(int)

    if date_to_timestamp:
        df = change_date_to_timestamp(df)
        df = df.drop(columns=["created_at"])

    df = add_exercise_code_level_lesson(df)
    try:
        df = df.drop(columns=["learning_object"])
    except:
        pass

    # Label encoding of student, skill (= exercise_code) and item (=exercise_code_level_lesson)
    student_label_encoder = LabelEncoder()
    df["student_id"] = student_label_encoder.fit_transform(df["student_id"].values.reshape(-1, 1))
    skill_label_encoder = LabelEncoder()
    df["skill_id"] = skill_label_encoder.fit_transform(df["exercise_code"].values.reshape(-1, 1))
    item_label_encoder = LabelEncoder()
    df["item_id"] = item_label_encoder.fit_transform(df["exercise_code_level_lesson"].values.reshape(-1, 1))

    return df[["timestamp", "student_id", "skill_id", "item_id", "correctness"]]


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
