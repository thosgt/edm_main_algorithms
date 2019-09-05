import pandas as pd

FEATURES_DEFAULT = ('lesson_id', 'exercise_code', 'learning_object', 'exercise_code_level')
TIME_WINDOWS_DEFAULT = ('60s', '1h', '1d', '5d', '30d', '365d')
COUNTERS_DEFAULT = ('attempts', 'wins')

def encode_df(df, features=FEATURES_DEFAULT, counters = COUNTERS_DEFAULT, time_windows=TIME_WINDOWS_DEFAULT):
    '''
    Get the wanted counters of each student and dummifies the categorical variables of the dataset.
    '''
    df = add_counters_for_all_features(df, features, counters, time_windows)
    df = pd.get_dummies(df, columns=['student_id', 'lesson_id', 'exercise_code', 'learning_object', 'exercise_code_level'])
    return df

def add_counters_for_all_features(df, features, counters, time_windows):
    '''
    Adds a column for all given features, all given counters of a student and all given time windows.
    '''
    for feature in features:
        df = add_feature_counters(df, feature, counters, time_windows)
    return df

def add_feature_counters(df, feature, counters, time_windows):
    '''
    For a given feature, adds a column for all given counters of a student and for all given time windows.
    '''
    for time_window in time_windows:
        for counter in counters:
            df = add_feature_past_counter_in_time_window(df, feature, counter, time_window)
    return df

def add_feature_past_counter_in_time_window(df, feature, counter, time_window):
    '''
    For a given feature, adds a column with the given counter of a student in the specified time_window
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
    if counter == "wins":
        counter_in_the_time_window = counter_in_the_time_window.sum()
    sorted_counter = counter_in_the_time_window.reset_index().fillna(0).sort_values(by=['timestamp', 'correctness'])
    df_copy[f'{feature}_{counter}_in_the_past_{str(time_window)}'] = sorted_counter['correctness'].values
    return df_copy.reset_index()

# TODO add tests