import pandas as pd

from pandas.testing import assert_frame_equal
from encode import add_one_feature_counter_in_one_time_window, add_feature_counters_in_one_time_window

# What I noticed : 
# - tests won't pass if there is only one student, I don't know if is good. I think it is caused by the groupby function
# - timestamp must be different from student to student or else the sorting by counter value and timestamp yields a false result
def test_add_one_feature_wins_in_one_time_window():
    feature = 'lesson_id'
    counter = 'wins'
    time_window = '1d'
    df = pd.DataFrame({'timestamp': ['2019-03-01 00:00:01', '2019-03-01 00:00:02', '2019-03-01 00:00:03', '2019-03-01 00:00:04'],
                       'student_id': [1, 1, 1, 2], 
                       'lesson_id': [101] * 4, 
                       'correctness': [1, 0, 1, 1]})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_wins = add_one_feature_counter_in_one_time_window(
        df, feature=feature, counter=counter, time_window=time_window)
    expected_df_with_counter = pd.DataFrame({'timestamp': ['2019-03-01 00:00:01', '2019-03-01 00:00:02', '2019-03-01 00:00:03', '2019-03-01 00:00:04'],
                                'student_id': [1, 1, 1, 2], 
                                'lesson_id': [101] * 4, 
                                'correctness': [1, 0, 1, 1],
                                f'{feature}_{counter}_in_the_past_{time_window}': [0.0, 1.0, 1.0, 0.0]})
    expected_df_with_counter["timestamp"] = pd.to_datetime(expected_df_with_counter["timestamp"])
    assert_frame_equal(df_wins, expected_df_with_counter)

def test_add_one_feature_attempts_in_one_time_window():
    feature = 'lesson_id'
    counter = 'attempts'
    time_window = '1d'
    df = pd.DataFrame({'timestamp': ['2019-03-01 00:00:01', '2019-03-01 00:00:02', '2019-03-01 00:00:03', '2019-03-01 00:00:04'],
                       'student_id': [1, 1, 1, 2], 
                       'lesson_id': [101] * 4, 
                       'correctness': [1, 0, 1, 1]})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_wins = add_one_feature_counter_in_one_time_window(
        df, feature=feature, counter=counter, time_window=time_window)
    expected_df_with_counter = pd.DataFrame({'timestamp': ['2019-03-01 00:00:01', '2019-03-01 00:00:02', '2019-03-01 00:00:03', '2019-03-01 00:00:04'],
                                'student_id': [1, 1, 1, 2], 
                                'lesson_id': [101] * 4, 
                                'correctness': [1, 0, 1, 1],
                                f'{feature}_{counter}_in_the_past_{time_window}': [0.0, 1.0, 2.0, 0.0]})
    expected_df_with_counter["timestamp"] = pd.to_datetime(expected_df_with_counter["timestamp"])
    assert_frame_equal(df_wins, expected_df_with_counter)

def test_add_one_feature_counters_in_one_time_window():
    feature = 'lesson_id'
    counters = ('attempts', 'wins')
    time_window = '1d'
    df = pd.DataFrame({'timestamp': ['2019-03-01 00:00:01', '2019-03-01 00:00:02', '2019-03-01 00:00:03', '2019-03-01 00:00:04'],
                       'student_id': [1, 1, 1, 2], 
                       'lesson_id': [101] * 4, 
                       'correctness': [1, 0, 1, 1]})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_with_counter = add_feature_counters_in_one_time_window(df, feature=feature, counters=counters, time_window=time_window)
    expected_df_with_counter = pd.DataFrame({'timestamp': ['2019-03-01 00:00:01', '2019-03-01 00:00:02', '2019-03-01 00:00:03', '2019-03-01 00:00:04'],
                                'student_id': [1, 1, 1, 2], 
                                'lesson_id': [101] * 4, 
                                'correctness': [1, 0, 1, 1],
                                f'{feature}_{counters[0]}_in_the_past_{time_window}': [0.0, 1.0, 2.0, 0.0],
                                f'{feature}_{counters[1]}_in_the_past_{time_window}': [0.0, 1.0, 1.0, 0.0]})
    expected_df_with_counter["timestamp"] = pd.to_datetime(expected_df_with_counter["timestamp"])
    assert_frame_equal(df_with_counter, expected_df_with_counter)