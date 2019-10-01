import os
import argparse
import pandas as pd
import numpy as np

from scipy import sparse
from scipy.sparse import csr_matrix, hstack, vstack
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


COUNTERS = ("attempts", "wins")


def encode_df(df, Q_mat, skill_counters=True):
    """Build sparse dataset from dense dataset and q-matrix.

    Arguments:
        df (pandas DataFrame): output by prepare_data.py
        Q_mat (sparse array): q-matrix, output by prepare_data.py
        skill_counters: if we want to include the counters of skill as well

    Output:
        sparse_df (sparse array): sparse dataset where first 4 columns are the same as in df
    """
    onehot_items = OneHotEncoder(categories="auto")  # to stop warnings
    onehot_items.fit(df["item_id"].values.reshape(-1, 1))

    features = None
    for user_id in tqdm(df["user_id"].unique()):
        df_user = df[df["user_id"] == user_id]
        user_features = encode_user_ffw(
            df_user,
            onehot_items=onehot_items,
            Q_mat=Q_mat,
            skill_counters=skill_counters,
        )
        user_features = hstack((df_user.values, user_features))
        features = (
            user_features if features is None else vstack((features, user_features))
        )
    return features


def encode_user_ffw(df_user, Q_mat, onehot_items, skill_counters=True):
    nb_user_exercises = len(df_user)

    labels = csr_matrix(df_user["correct"].values.reshape(-1, 1))
    item_ids = df_user["item_id"].values.reshape(-1, 1)
    item_ids_onehot = onehot_items.transform(item_ids)

    skill_ids_onehot = Q_mat[item_ids.flatten()]

    all_counters = csr_matrix((nb_user_exercises, 0))
    for counter in COUNTERS:
        user_item_counter = get_user_counter(item_ids_onehot, labels, counter=counter)
        all_counters = hstack((all_counters, user_item_counter))
        if skill_counters:
            user_skill_counter = get_user_counter(
                skill_ids_onehot, labels, counter=counter
            )
            all_counters = hstack((all_counters, user_skill_counter))
    return all_counters


def get_user_counter(feature_id_onehot, labels, counter):
    array_to_accumulate = feature_id_onehot.toarray()
    if counter == "attempts":
        pass
    elif counter == "wins":
        array_to_accumulate *= labels.toarray()
    counts = accumulate(array_to_accumulate)
    counter = phi(counts)
    return counter


def accumulate(x):
    return vstack((csr_matrix((1, x.shape[1])), csr_matrix(np.cumsum(x, 0))))[:-1]


def phi(x):
    return x.log1p()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode feature matrix for feedforward network baseline."
    )
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--n_traces", type=int, default=500000)

    args = parser.parse_args()

    data_path = os.path.join("data", args.dataset)

    df = pd.read_csv(os.path.join(data_path, "preprocessed_data.csv"), sep="\t")
    Q_mat = sparse.load_npz(os.path.join(data_path, "q_mat.npz"))
    X = encode_df(df[-args.n_traces :], Q_mat)
    sparse.save_npz(os.path.join(data_path, f"X-ffw-{args.n_traces}-traces"), X)

