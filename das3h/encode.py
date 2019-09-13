import pandas as pd
import numpy as np

from typing import Optional
from tqdm import tqdm

TIME_WINDOWS_DEFAULT = ("60s", "1h", "1d", "5d", "30d", "365d")
COUNTERS_DEFAULT = ("attempts", "wins")


def encode_df(
    df: pd.DataFrame,
    exercises: Optional[tuple] = None,
    counters: tuple = COUNTERS_DEFAULT,
    time_windows: tuple = TIME_WINDOWS_DEFAULT,
) -> pd.DataFrame:
    """
    Get the wanted counters of each student and
    dummifies the categorical variables of the dataset.
    """
    if exercises is None:
        exercises = set(df["exercise_code"].unique())
    # exercises done by only one student must be removed
    exercises_to_keep = remove_exercises_done_by_only_one_student(df, exercises)
    df = add_counters_for_all_exercises(
        df, exercises_to_keep, counters, time_windows
    ).fillna(0)
    df = pd.get_dummies(
        df, columns=["student_id", "exercise_code_level_lesson", "exercise_code"]
    )
    return df


def remove_exercises_done_by_only_one_student(df, exercises):
    """
    Groupby function is not working with them for now
    """
    nb_student_by_exo = (
        df[["student_id", "exercise_code"]]
        .drop_duplicates()
        .groupby("exercise_code", as_index=False)
        .count()
    )
    exercises_to_keep = exercises
    exercises_to_remove = nb_student_by_exo[nb_student_by_exo["student_id"] == 1][
        "exercise_code"
    ].values
    for exercise in exercises_to_remove:
        exercises_to_keep.remove(exercise)
    return exercises_to_keep


def add_counters_for_all_exercises(
    df: pd.DataFrame, exercises: tuple, counters: tuple, time_windows: tuple
) -> pd.DataFrame:
    """
    Adds a column for all given exercises, all given counters
    of a student and all given time windows.
    """
    for exercise_code in tqdm(
        exercises, desc=f"Adding the counters for {len(exercises)} exercises"
    ):
        df = add_exercise_code_counters_for_each_time_window(
            df, exercise_code, counters, time_windows
        )
    return df


def add_exercise_code_counters_for_each_time_window(
    df: pd.DataFrame, exercise_code: str, counters: tuple, time_windows: tuple
) -> pd.DataFrame:
    """
    For a given exercise_code, adds a column for all given counters
    of a student and for all given time windows.
    """
    for time_window in tqdm(
        time_windows,
        desc=f"Adding the counters for the time windows of {exercise_code}",
    ):
        df = add_exercise_code_counters_in_one_time_window(
            df, exercise_code, counters, time_window
        )
    return df


def add_exercise_code_counters_in_one_time_window(
    df: pd.DataFrame, exercise_code: str, counters: tuple, time_window: str
) -> pd.DataFrame:
    """
    For a given exercise_code, adds a column for all given counters
    of a student and for one given time window.
    """
    for counter in counters:
        df = add_one_exercise_code_counter_in_one_time_window(
            df, exercise_code, counter, time_window
        )
    return df


def add_one_exercise_code_counter_in_one_time_window(
    df: pd.DataFrame, exercise_code: str, counter: str, time_window: str
) -> pd.DataFrame:
    df_copy = df.copy()
    filtered_df = df[df["exercise_code"] == exercise_code]
    filtered_df_and_timestamp_index = filtered_df.set_index("timestamp")
    counter_in_the_time_window = filtered_df_and_timestamp_index.groupby(
        by=["student_id"], as_index=False
    ).rolling(time_window, closed="left")["correctness"]
    assert counter in ("wins", "attempts")
    if counter == "attempts":
        counter_in_the_time_window = counter_in_the_time_window.count()
    elif counter == "wins":
        counter_in_the_time_window = counter_in_the_time_window.sum()
    exercise_code_counter = (
        counter_in_the_time_window.reset_index()
        .fillna(0)
        .sort_values(by=["timestamp", "correctness"])
    )
    exercise_code_counter["index"] = filtered_df.index
    exercise_code_counter = exercise_code_counter.set_index("index")
    df_copy[f"{exercise_code}_{counter}_in_the_past_{time_window}"] = scaling_function(
        exercise_code_counter["correctness"]
    )
    return df_copy


def scaling_function(x, how="log"):
    if how == "log":
        return np.log(1 + x)
    else:
        return np.nan
