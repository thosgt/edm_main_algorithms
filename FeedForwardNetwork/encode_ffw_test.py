import pandas as pd
import numpy as np

from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal
from encode_ffw import encode_student_ffw, get_student_counter, encode_df
from sklearn.preprocessing import OneHotEncoder


def test_attempts_counter():
    df_exercise_tuple = pd.DataFrame(
        {
            "phono_3_lesson_102": [1, 1, 0, 1, 0, 1, 1],
            "phono_3_lesson_103": [0, 0, 1, 0, 1, 0, 0],
            "correctness": [0, 1, 1, 0, 1, 0, 1],
        }
    )
    feature_id_onehot = df_exercise_tuple[
        ["phono_3_lesson_102", "phono_3_lesson_103"]
    ].values
    labels = df_exercise_tuple["correctness"].values.reshape(-1, 1)
    counter = "attempts"
    student_attempts = get_student_counter(feature_id_onehot, labels, counter)
    expected_array = np.log(
        1 + np.array([[0, 0], [1, 0], [2, 0], [2, 1], [3, 1], [3, 2], [4, 2]])
    )
    assert_array_equal(student_attempts, expected_array)


def test_wins_counter():
    df_exercise_tuple = pd.DataFrame(
        {
            "phono_3_lesson_102": [1, 1, 0, 1, 0, 1, 1],
            "phono_3_lesson_103": [0, 0, 1, 0, 1, 0, 0],
            "correctness": [0, 1, 1, 0, 1, 0, 1],
        }
    )
    feature_id_onehot = df_exercise_tuple[
        ["phono_3_lesson_102", "phono_3_lesson_103"]
    ].values
    labels = df_exercise_tuple["correctness"].values.reshape(-1, 1)
    counter = "wins"
    student_attempts = get_student_counter(feature_id_onehot, labels, counter)
    expected_array = np.log(
        1 + np.array([[0, 0], [0, 0], [1, 0], [1, 1], [1, 1], [1, 2], [1, 2]])
    )
    assert_array_equal(student_attempts, expected_array)


def test_encoding_counter():
    df_student = pd.DataFrame(
        {
            "item_id": [0, 0, 1, 0, 1, 0, 0],
            "skill_id": [0, 0, 0, 0, 0, 0, 0],
            "correctness": [0, 1, 1, 0, 1, 0, 1],
        }
    )
    onehot_items = OneHotEncoder()
    onehot_skills = OneHotEncoder()
    onehot_items.fit(df_student["item_id"].values.reshape(-1, 1))
    onehot_skills.fit(df_student["skill_id"].values.reshape(-1, 1))

    student_ffw_encoding = encode_student_ffw(
        df_student,
        onehot_items=onehot_items,
        onehot_skills=onehot_skills,
        skill_counters=True,
    )
    expected_attempts_array = np.array(
        [[0, 0, 0], [1, 0, 1], [2, 0, 2], [2, 1, 3], [3, 1, 4], [3, 2, 5], [4, 2, 6]]
    )
    expected_wins_array = np.array(
        [[0, 0, 0], [0, 0, 0], [1, 0, 1], [1, 1, 2], [1, 1, 2], [1, 2, 3], [1, 2, 3]]
    )
    expected_array = np.concatenate(
        (expected_attempts_array, expected_wins_array), axis=1
    )
    expected_array = np.log(1 + expected_array)
    assert_array_equal(student_ffw_encoding, expected_array)


def test_encoding_counter_two_students():
    df_student = pd.DataFrame(
        {
            "student_id": [0, 0, 1, 0, 1, 0, 0],
            "item_id": [0, 0, 1, 0, 1, 0, 0],
            "skill_id": [0, 0, 0, 0, 0, 0, 0],
            "correctness": [0, 1, 1, 0, 1, 0, 1],
        }
    )
    student_ffw_encoding = encode_df(df_student, skill_counters=False).toarray()
    expected_attempts_array_0 = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
    expected_wins_array_0 = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [1, 0]])
    expected_attempts_array_1 = np.array([[0, 0], [0, 1]])
    expected_wins_array_1 = np.array([[0, 0], [0, 1]])
    expected_array_0 = np.concatenate(
        (expected_attempts_array_0, expected_wins_array_0), axis=1
    )
    expected_array_1 = np.concatenate(
        (expected_attempts_array_1, expected_wins_array_1), axis=1
    )
    expected_array = np.concatenate((expected_array_0, expected_array_1), axis=0)
    expected_array = np.log(1 + expected_array)

    expected_array = np.hstack(
        (df_student.sort_values(by="student_id").values, expected_array)
    )
    assert_array_equal(student_ffw_encoding, expected_array)
