import pandas as pd
import numpy as np

from pandas.testing import assert_frame_equal
from clean import clean_df

def test_that_clean_df_cleans_properly():
    df = pd.DataFrame({'Unnamed: 0': [1, 2, 3], 'id': [1, 2, 3],
                       'created_at': ['2019-03-01 00:00:01', '2019-03-01 00:00:02', '2019-03-01 00:00:03'],
                       'student_id': [0, 0, 0], 
                       'exercise_code': ['grapho', 'phono', 'discovery'], 
                       'level': [3, 4, 1],
                       'correctness': [True, False, None]})

    cleaned_df = clean_df(df)
    expected_df = pd.DataFrame({'timestamp': ['2019-03-01 00:00:01', '2019-03-01 00:00:02'],
                       'student_id': [0, 0], 
                       'exercise_code': ['grapho', 'phono'],
                       'exercise_code_level': ['grapho_3', 'phono_4'], 
                       'correctness': [1, 0]})
    expected_df["timestamp"] = pd.to_datetime(expected_df["timestamp"])
    assert_frame_equal(cleaned_df, expected_df, check_like=True) # ignore column order