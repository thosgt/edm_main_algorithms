import pandas as pd
import numpy as np

from pandas.testing import assert_frame_equal
from encode import scaling_function, add_one_exercise_code_counter_in_one_time_window

# What I noticed : 
# - tests won't pass if there is only one student, I don't know if is good. I think it is caused by the groupby function
# - timestamp must be different from student to student or else the sorting by counter value and timestamp yields a false result

def test_add_one_exercise_code_wins_in_one_time_window():
    exercise_code = 'phono'
    time_window = '1d'
    counter = 'wins'
    df = pd.DataFrame({'timestamp': ['2019-03-01 00:00:01', '2019-03-01 00:00:02', '2019-03-01 00:00:03', '2019-03-01 00:00:04', '2019-03-01 00:00:05'],
                    'student_id': [1, 1, 2, 2, 1], 
                    'exercise_code': ['phono', 'phono', 'phono', 'grapho', 'phono'], 
                    'correctness': [1, 0, 1, 1, 1]})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_with_one_more_column = add_one_exercise_code_counter_in_one_time_window(df, exercise_code, counter, time_window)
    expected_df = pd.DataFrame({'timestamp': ['2019-03-01 00:00:01', '2019-03-01 00:00:02', '2019-03-01 00:00:03', '2019-03-01 00:00:04', '2019-03-01 00:00:05'],
                    'student_id': [1, 1, 2, 2, 1], 
                    'exercise_code': ['phono', 'phono', 'phono', 'grapho', 'phono'], 
                    'correctness': [1, 0, 1, 1, 1],
                    'phono_wins_in_the_past_1d': [0, 1, 0, None, 1]})
    expected_df['phono_wins_in_the_past_1d'] = expected_df['phono_wins_in_the_past_1d'].apply(scaling_function)
    expected_df["timestamp"] = pd.to_datetime(expected_df["timestamp"])
    assert_frame_equal(expected_df, df_with_one_more_column, check_like=True)

def test_add_one_exercise_code_attempts_in_one_time_window():
    exercise_code = 'phono'
    time_window = '1d'
    counter = 'attempts'
    df = pd.DataFrame({'timestamp': ['2019-03-01 00:00:01', '2019-03-01 00:00:02', '2019-03-01 00:00:03', '2019-03-01 00:00:04', '2019-03-01 00:00:05'],
                    'student_id': [1, 1, 2, 2, 1], 
                    'exercise_code': ['phono', 'phono', 'phono', 'grapho', 'phono'], 
                    'correctness': [1, 0, 1, 1, 1]})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_with_one_more_column = add_one_exercise_code_counter_in_one_time_window(df, exercise_code, counter, time_window)
    expected_df = pd.DataFrame({'timestamp': ['2019-03-01 00:00:01', '2019-03-01 00:00:02', '2019-03-01 00:00:03', '2019-03-01 00:00:04', '2019-03-01 00:00:05'],
                    'student_id': [1, 1, 2, 2, 1], 
                    'exercise_code': ['phono', 'phono', 'phono', 'grapho', 'phono'], 
                    'correctness': [1, 0, 1, 1, 1],
                    'phono_attempts_in_the_past_1d': [0, 1, 0, None, 2]})
    expected_df['phono_attempts_in_the_past_1d'] = expected_df['phono_attempts_in_the_past_1d'].apply(scaling_function)
    expected_df["timestamp"] = pd.to_datetime(expected_df["timestamp"])
    assert_frame_equal(expected_df, df_with_one_more_column, check_like=True)

def test_add_one_exercise_code_wins_only_use_traces_in_the_given_time_window():
    exercise_code = 'phono'
    time_window = '1d'
    counter = 'wins'
    df = pd.DataFrame({'timestamp': ['2019-03-01 00:00:01', '2019-03-01 00:00:02', '2019-03-01 00:00:03', '2019-03-01 00:00:04', '2019-03-03 00:00:05'],
                    'student_id': [1, 1, 2, 2, 1], 
                    'exercise_code': ['phono', 'phono', 'phono', 'grapho', 'phono'], 
                    'correctness': [1, 0, 1, 1, 1]})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_with_one_more_column = add_one_exercise_code_counter_in_one_time_window(df, exercise_code, counter, time_window)
    expected_df = pd.DataFrame({'timestamp': ['2019-03-01 00:00:01', '2019-03-01 00:00:02', '2019-03-01 00:00:03', '2019-03-01 00:00:04', '2019-03-03 00:00:05'],
                    'student_id': [1, 1, 2, 2, 1], 
                    'exercise_code': ['phono', 'phono', 'phono', 'grapho', 'phono'], 
                    'correctness': [1, 0, 1, 1, 1],
                    'phono_wins_in_the_past_1d': [0, 1, 0, None, 0]})
    expected_df['phono_wins_in_the_past_1d'] = expected_df['phono_wins_in_the_past_1d'].apply(scaling_function)
    expected_df["timestamp"] = pd.to_datetime(expected_df["timestamp"])
    assert_frame_equal(expected_df, df_with_one_more_column, check_like=True)