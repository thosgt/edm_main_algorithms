import pandas as pd


def print_cleaned_df_with_information(cleaned_df: pd.DataFrame):
    nb_students = len(cleaned_df["student_id"].unique())
    print(f"There are {nb_students} students in this dataset.")
    print()
    trace_repartition = (
        cleaned_df.groupby("exercise_code")
        .count()
        .rename(columns={"correctness": "traces_count"})["traces_count"]
        .sort_values(ascending=False)
    )
    print("This is the repartition of traces:")
    print(trace_repartition.plot(kind="bar"))


