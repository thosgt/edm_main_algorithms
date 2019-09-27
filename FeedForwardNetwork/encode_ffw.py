import os
import argparse
from scipy import sparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder


COUNTERS = ("attempts", "wins")


def encode_df(df, skill_counters=True):
    onehot_items = OneHotEncoder()
    onehot_items.fit(df["item_id"].values.reshape(-1, 1))
    onehot_skills = OneHotEncoder()
    onehot_skills.fit(df["skill_id"].values.reshape(-1, 1))

    features = None
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id]
        user_features = encode_user_ffw(
            df_user,
            onehot_items=onehot_items,
            onehot_skills=onehot_skills,
            skill_counters=skill_counters,
        )
        user_features = np.hstack((df_user.values, user_features))
        features = (
            user_features
            if features is None
            else np.vstack((features, user_features))
        )
    return sparse.csr_matrix(features)


def encode_user_ffw(df_user, onehot_items, onehot_skills, skill_counters=True):
    nb_user_exercises = len(df_user)

    labels = df_user["correct"].values.reshape(-1, 1)
    item_ids = df_user["item_id"].values.reshape(-1, 1)
    item_ids_onehot = onehot_items.transform(item_ids).toarray()

    skill_ids = df_user["skill_id"].values.reshape(-1, 1)
    skill_ids_onehot = onehot_skills.transform(skill_ids).toarray()

    all_counters = np.empty((nb_user_exercises, 0))
    for counter in COUNTERS:
        user_item_counter = get_user_counter(
            item_ids_onehot, labels, counter=counter
        )
        all_counters = np.hstack((all_counters, user_item_counter))
        if skill_counters:
            user_skill_counter = get_user_counter(
                skill_ids_onehot, labels, counter=counter
            )
            all_counters = np.hstack((all_counters, user_skill_counter))
    return all_counters


def get_user_counter(feature_id_onehot, labels, counter):
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
    sparse.save_npz(os.path.join(data_path, f"X-ffw"), X)

