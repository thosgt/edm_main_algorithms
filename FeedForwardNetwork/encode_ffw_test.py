import pandas as pd
import numpy as np
from scipy import sparse

from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal
from encode_ffw import encode_user_ffw, get_user_counter, encode_df
from sklearn.preprocessing import OneHotEncoder


def test_attempts_counter():
    df_exercise_tuple = pd.DataFrame(
        {
            "phono_3_lesson_102": [1, 1, 0, 1, 0, 1, 1],
            "phono_3_lesson_103": [0, 0, 1, 0, 1, 0, 0],
            "correct": [0, 1, 1, 0, 1, 0, 1],
        }
    )
    feature_id_onehot = sparse.csr_matrix(
        df_exercise_tuple[["phono_3_lesson_102", "phono_3_lesson_103"]].values
    )
    labels = sparse.csr_matrix(df_exercise_tuple["correct"].values.reshape(-1, 1))
    counter = "attempts"
    user_attempts = get_user_counter(feature_id_onehot, labels, counter).toarray()
    expected_array = np.log(
        1 + np.array([[0, 0], [1, 0], [2, 0], [2, 1], [3, 1], [3, 2], [4, 2]])
    )
    assert_array_equal(user_attempts, expected_array)


def test_wins_counter():
    df_exercise_tuple = pd.DataFrame(
        {
            "phono_3_lesson_102": [1, 1, 0, 1, 0, 1, 1],
            "phono_3_lesson_103": [0, 0, 1, 0, 1, 0, 0],
            "correct": [0, 1, 1, 0, 1, 0, 1],
        }
    )
    feature_id_onehot = sparse.csr_matrix(
        df_exercise_tuple[["phono_3_lesson_102", "phono_3_lesson_103"]].values
    )
    labels = sparse.csr_matrix(df_exercise_tuple["correct"].values.reshape(-1, 1))
    counter = "wins"
    user_attempts = get_user_counter(feature_id_onehot, labels, counter).toarray()
    expected_array = np.log(
        1 + np.array([[0, 0], [0, 0], [1, 0], [1, 1], [1, 1], [1, 2], [1, 2]])
    )
    assert_array_equal(user_attempts, expected_array)


def test_encoding_counter():
    df_user = pd.DataFrame(
        {
            "item_id": [0, 0, 1, 0, 1, 0, 0],
            "skill_id": [0, 0, 0, 0, 0, 0, 0],
            "correct": [0, 1, 1, 0, 1, 0, 1],
        }
    )
    Q_mat = sparse.csr_matrix([[1], [1]])
    onehot_items = OneHotEncoder()
    onehot_items.fit(df_user["item_id"].values.reshape(-1, 1))

    user_ffw_encoding = encode_user_ffw(
        df_user, onehot_items=onehot_items, Q_mat=Q_mat, skill_counters=True
    ).toarray()
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
    assert_array_equal(user_ffw_encoding, expected_array)


def test_encoding_counter_two_users():
    df_user = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 0, 1, 0, 0],
            "item_id": [0, 0, 1, 0, 1, 0, 0],
            "skill_id": [0, 0, 0, 0, 0, 0, 0],
            "correct": [0, 1, 1, 0, 1, 0, 1],
        }
    )
    Q_mat = sparse.csr_matrix([[1], [1]])  # useless here as skill_counters=False
    user_ffw_encoding = encode_df(df_user, Q_mat, skill_counters=False).toarray()
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
        (df_user.sort_values(by="user_id").values, expected_array)
    )
    assert_array_equal(user_ffw_encoding, expected_array)
