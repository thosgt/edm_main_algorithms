import pandas as pd
import numpy as np


def get_coefs_in_dataframe(model, X: pd.DataFrame):
    return (
        pd.DataFrame(data=model.coef_[0], index=X.columns)
        .reset_index()
        .rename(columns={"index": "columns", 0: "coefs"})
    )


def get_students_alphas(coefs: pd.DataFrame):
    return coefs[coefs["columns"].str.contains("student_id")].sort_values("coefs")


def get_exercise_code_betas(coefs: pd.DataFrame):
    exercise_codes = [
        f"exercise_code_{exercise_code}"
        for exercise_code in get_available_exercise_codes(coefs)
    ]
    return coefs[coefs["columns"].isin(exercise_codes)].sort_values("coefs")


def get_gamma_of_exercise(
    coefs: pd.DataFrame, exercise_code: str, level: int, lesson: int
):
    return coefs[
        coefs["columns"].str.contains(
            f"{exercise_code}_{str(level)}_lesson_{str(lesson)}"
        )
    ].sort_values("coefs")


def get_available_exercise_codes(coefs: pd.DataFrame):
    return np.array(
        list(
            map(
                lambda x: x[len("exercise_code_") :],
                coefs[coefs["columns"].str.startswith("exercise_code")][
                    ~coefs["columns"].str.startswith("exercise_code_level_lesson")
                ]["columns"].values,
            )
        )
    )


def get_available_exercise_code_level_lesson_tuples(coefs: pd.DataFrame):
    return np.array(
        list(
            map(
                lambda x: x[len("exercise_code_level_lesson_") :],
                coefs[coefs["columns"].str.startswith("exercise_code_level_lesson")][
                    "columns"
                ].values,
            )
        )
    )


def get_exercise_gammas_of_one_exercise_code(coefs: pd.DataFrame, exercise_code: str):
    available_exercise_codes = get_available_exercise_codes(coefs)
    if exercise_code not in available_exercise_codes:
        print('Error : exercise_code not in the dataset')
        return
    exercise_code_level_lesson_tuples = get_available_exercise_code_level_lesson_tuples(
        coefs
    )
    filtered_tuples = [
        f"exercise_code_level_lesson_{tuple}"
        for tuple in exercise_code_level_lesson_tuples
        if exercise_code in tuple
    ]
    return coefs[coefs["columns"].isin(filtered_tuples)].sort_values(
        by="coefs", ascending=False
    )


def get_thetas_of_one_exercise_code(coefs: pd.DataFrame, exercise_code: str):
    return coefs[coefs["columns"].str.startswith(exercise_code)]
