import pandas as pd
import numpy as np

from tqdm import tqdm

FEATURES_DEFAULT = ('lesson_id', 'exercise_code', 'learning_object', 'exercise_code_level')
TIME_WINDOWS_DEFAULT = ('60s', '1h', '1d', '5d', '30d', '365d')
COUNTERS_DEFAULT = ('attempts', 'wins')

def encode_df(df: pd.DataFrame, features: tuple=FEATURES_DEFAULT, counters: tuple=COUNTERS_DEFAULT, time_windows: tuple=TIME_WINDOWS_DEFAULT)-> pd.DataFrame:
    '''
    Get the wanted counters of each student and dummifies the categorical variables of the dataset.
    '''
    df = add_counters_for_all_features(df, features, counters, time_windows)
    df = pd.get_dummies(df, columns=['student_id', 'lesson_id', 'exercise_code', 'learning_object', 'exercise_code_level'])
    return df

def add_counters_for_all_features(df: pd.DataFrame, features: tuple, counters: tuple, time_windows: tuple)-> pd.DataFrame:
    '''
    Adds a column for all given features, all given counters of a student and all given time windows.
    '''
    for feature in tqdm(features, desc=f'Adding the counters for {len(features)} features'):
        df = add_feature_counters_for_each_time_window(df, feature, counters, time_windows)
    return df

def add_feature_counters_for_each_time_window(df: pd.DataFrame, feature: str, counters: tuple, time_windows: tuple)-> pd.DataFrame:
    '''
    For a given feature, adds a column for all given counters of a student and for all given time windows.
    '''
    for time_window in tqdm(time_windows, desc=f'Adding the counters for the time windows of {feature}'):
        df = add_feature_counters_in_one_time_window(df, feature, counters, time_window)
    return df

def add_feature_counters_in_one_time_window(df: pd.DataFrame, feature: str, counters: tuple, time_window: str)-> pd.DataFrame:
    '''
    For a given feature, adds a column for all given counters of a student and for one given time window.
    '''
    for counter in counters:
        df = add_one_feature_counter_in_one_time_window(df, feature, counter, time_window)
    return df

def add_one_feature_counter_in_one_time_window(df: pd.DataFrame, feature: str, counter: str, time_window: str)-> pd.DataFrame:
    '''
    For a given feature, adds a column with the given counter of a student in the specified time window.
    Possible counters :
    - wins
    - attempts
    '''
    df_copy = df.copy()
    df_copy = df_copy.set_index('timestamp')
    # rolling works properly on offset only if the timestamp is the index
    counter_in_the_time_window = df_copy.groupby(by=['student_id', feature], as_index=False).rolling(time_window, closed='left')['correctness']
    # closed='left' removes the current row from the count and thus prevent data leakage. Think of [-3,0[
    if counter == "attempts":
        counter_in_the_time_window = counter_in_the_time_window.count()
    elif counter == "wins":
        counter_in_the_time_window = counter_in_the_time_window.sum()
    sorted_counter = counter_in_the_time_window.reset_index().fillna(0).sort_values(by=['timestamp', 'correctness'])
    df_copy[f'{feature}_{counter}_in_the_past_{str(time_window)}'] = sorted_counter['correctness'].apply(scaling_function).values
    return df_copy.reset_index()

def scaling_function(x, type='log'):
    if type == 'log':
        return np.log(1 + x)