import pandas as pd

from prepare_sequences import prepare_df
from pandas.testing import assert_frame_equal

df = pd.DataFrame(
    {
        "student_id": [1, 2, 3, 1, 5, 5, 4],
        "exercise_code_level_lesson": [
            "phono_3_lesson_102",
            "phono_3_lesson_101",
            "phono_3_lesson_103",
            "phono_3_lesson_101",
            "phono_3_lesson_102",
            "phono_3_lesson_103",
            "phono_3_lesson_103",
        ],
        "correctness": [0, 1, 1, 0, 1, 0, 1],
    }
)


def test_that_cleaning_of_df_works():
    prepared_df, _ = prepare_df(df)
    expected_prepared_df = pd.DataFrame(
        {
            "student_id": {0: 1, 1: 2, 2: 3, 3: 1, 4: 5, 5: 5, 6: 4},
            "correctness": {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1},
            "exercise_code_level_lesson": {0: 1, 1: 0, 2: 2, 3: 0, 4: 1, 5: 2, 6: 2},
            "concat_exercise_correctness_0_0": {
                0: 0,
                1: 0,
                2: 0,
                3: 1,
                4: 0,
                5: 0,
                6: 0,
            },
            "concat_exercise_correctness_0_1": {
                0: 0,
                1: 1,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
            },
            "concat_exercise_correctness_1_0": {
                0: 1,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
            },
            "concat_exercise_correctness_1_1": {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 1,
                5: 0,
                6: 0,
            },
            "concat_exercise_correctness_2_0": {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 1,
                6: 0,
            },
            "concat_exercise_correctness_2_1": {
                0: 0,
                1: 0,
                2: 1,
                3: 0,
                4: 0,
                5: 0,
                6: 1,
            },
        }
    )
    assert_frame_equal(prepared_df, expected_prepared_df, check_dtype=False, check_like=True)

