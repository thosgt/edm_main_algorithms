import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


def prepare_df(df: pd.DataFrame)-> pd.DataFrame:
    label_encoder = LabelEncoder()
    df_copy = df.copy()
    df_copy["exercise_code_level_lesson"] = label_encoder.fit_transform(
        df_copy["exercise_code_level_lesson"]
    )
    df_copy["concat_exercise_correctness"] = (
        df_copy["exercise_code_level_lesson"].map(str)
        + "_"
        + df_copy["correctness"].map(str)
    )
    df_to_feed_network = pd.get_dummies(
        df_copy[
            ["correctness", "student_id", "exercise_code_level_lesson", "concat_exercise_correctness"]
        ],
        columns=["concat_exercise_correctness"],
        sparse=True,
    )
    n_expected_columns = df_copy.exercise_code_level_lesson.unique().shape[0]
    expected_columns = []
    for i in range(n_expected_columns):
        for correctness in (0, 1):
            expected_columns.append(f'concat_exercise_correctness_{i}_{correctness}')
    for column in expected_columns:
        if column not in df_to_feed_network.columns:
            df_to_feed_network[column] = 0
    return df_to_feed_network, label_encoder


def prepare_sequences(df: pd.DataFrame):
    # idea have a generator to spare memory ? no if several epochs
    # idea add shuffling somewhere ?
    student_ids = df["student_id"].unique()
    exercise_sequences = []
    # le.save somewhere ?
    for student_id in student_ids:
        df_of_student = df[df["student_id"] == student_id].drop(columns=["student_id"])
        exercise_sequences.append(df_of_student)
    return exercise_sequences

