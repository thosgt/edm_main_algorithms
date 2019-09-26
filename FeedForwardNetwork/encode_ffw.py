import os
import argparse
from scipy import sparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder


COUNTERS = ("attempts", "wins")


def encode_df(df, skill_counters=True):
    nb_skills = len(df["skill_id"].unique())
    nb_items = len(df["item_id"].unique())
    features = None
    for student_id in df["student_id"].unique():
        df_student = df[df["student_id"] == student_id]
        student_features = encode_student_ffw(
            df_student,
            nb_skills=nb_skills,
            nb_items=nb_items,
            skill_counters=skill_counters,
        )
        features = (
            student_features
            if features is None
            else np.vstack((features, student_features))
        )
    return sparse.csr_matrix(features)


def encode_student_ffw(df_student, nb_items, nb_skills, skill_counters=True):
    onehot_items = OneHotEncoder(categories=[range(nb_items)])
    onehot_skills = OneHotEncoder(categories=[range(nb_skills)])

    nb_student_exercises = len(df_student)

    labels = df_student["correctness"].values.reshape(-1, 1)
    item_ids = df_student["item_id"].values.reshape(-1, 1)
    item_ids_onehot = onehot_items.fit_transform(item_ids).toarray()

    skill_ids = df_student["skill_id"].values.reshape(-1, 1)
    skill_ids_onehot = onehot_skills.fit_transform(skill_ids).toarray()

    all_counters = np.empty((nb_student_exercises, 0))
    for counter in COUNTERS:
        student_item_counter = get_student_counter(
            item_ids_onehot, labels, counter=counter
        )
        all_counters = np.hstack((all_counters, student_item_counter))
        if skill_counters:
            student_skill_counter = get_student_counter(
                skill_ids_onehot, labels, counter=counter
            )
            all_counters = np.hstack((all_counters, student_skill_counter))
    return all_counters


def get_student_counter(feature_id_onehot, labels, counter):
    array_to_accumulate = np.empty(feature_id_onehot.shape)
    if counter == "attempts":
        array_to_accumulate = feature_id_onehot
    elif counter == "wins":
        array_to_accumulate = feature_id_onehot * labels
    counts = accumulate(array_to_accumulate)
    counter = phi(counts)
    return counter


def accumulate(x):
    return np.vstack((np.zeros(x.shape[1]), np.cumsum(x, 0)))[:-1]


def phi(x):
    return np.log(1 + x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode feature matrix for feedforward network baseline."
    )
    parser.add_argument("--dataset", type=str)
 
    args = parser.parse_args()

    data_path = os.path.join("data", args.dataset)


    df = pd.read_csv(os.path.join(data_path, "preprocessed_data.csv"), sep="\t")
    X = encode_df(df)
    sparse.save_npz(
        os.path.join(
            data_path, f"X-ffw"
        ),
        X,
    )

