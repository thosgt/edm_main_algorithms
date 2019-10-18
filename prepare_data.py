import numpy as np
import pandas as pd
from scipy import sparse
import argparse
import os
from time import process_time


def prepare_assistments(data_name, min_interactions_per_user, remove_nan_skills):
    """Preprocess ASSISTments dataset.
    
    Arguments:
        data_name: "assistments09", "assistments12", "assistments15" or "assistments17"
        min_interactions_per_user (int): minimum number of interactions per student
        remove_nan_skills (bool): if True, remove interactions with no skill tag
    Outputs:
        df (pandas DataFrame): preprocessed ASSISTments dataset with user_id, item_id,
            timestamp and correct features
        Q_mat (item-skill relationships sparse array): corresponding q-matrix
    """
    data_path = os.path.join("data", data_name)
    df = pd.read_csv(os.path.join(data_path, "data.csv"), encoding="ISO-8859-1")

    # Only 2012 and 2017 versions have timestamps
    if data_name == "assistments09":
        df = df.rename(columns={"problem_id": "item_id"})
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assistments12":
        df = df.rename(columns={"problem_id": "item_id"})
        df = add_timestamp(df, "start_time")
    elif data_name == "assistments15":
        df = df.rename(columns={"sequence_id": "item_id"})
        df["skill_id"] = df["item_id"]
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assistments17":
        df = df.rename(
            columns={
                "startTime": "timestamp",
                "studentId": "user_id",
                "problemId": "item_id",
                "skill": "skill_id",
            }
        )
        df = add_timestamp(df, "timestamp")

    # Sort data temporally
    if data_name in ["assistments12", "assistments17"]:
        df.sort_values(by="timestamp", inplace=True)
    elif data_name == "assistments09":
        df.sort_values(by="order_id", inplace=True)
    elif data_name == "assistments15":
        df.sort_values(by="log_id", inplace=True)

    df = general_cleaning(df, min_interactions_per_user)
    df = remove_nan_skill(remove_nan_skills, df)

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
    df["skill_id"] = np.unique(df["skill_id"], return_inverse=True)[1]

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    # Remove row duplicates due to multiple skills for one item
    if data_name == "assistments09":
        df = df.drop_duplicates("order_id")
    elif data_name == "assistments17":
        df = df.drop_duplicates(["user_id", "timestamp"])

    # Get unique skill id from combination of all skill ids
    df["skill_id"] = np.unique(Q_mat, axis=0, return_inverse=True)[1][df["item_id"]]

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    df.reset_index(inplace=True, drop=True)

    # Save data
    save_data(df, data_path, Q_mat)


def remove_nan_skill(remove_nan_skills, df):
    # Filter nan skills
    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]
    else:
        df.ix[df["skill_id"].isnull(), "skill_id"] = -1
    return df


def prepare_kddcup10(
    data_name, min_interactions_per_user, kc_col_name, remove_nan_skills
):
    """Preprocess KDD Cup 2010 dataset.
    Arguments:
        data_name (str): "bridge_algebra06" or "algebra05"
        min_interactions_per_user (int): minimum number of interactions per student
        kc_col_name (str): Skills id column
        remove_nan_skills (bool): if True, remove interactions with no skill tag
    Outputs:
        df (pandas DataFrame): preprocessed KDD Cup 2010 dataset with user_id, item_id,
            timestamp and correct features
        Q_mat (item-skill relationships sparse array): corresponding q-matrix
    """
    data_path = os.path.join("data", data_name)
    df = pd.read_csv(os.path.join(data_path, "data.txt"), delimiter="\t")
    df = df.rename(
        columns={
            "Anon Student Id": "user_id",
            "Correct First Attempt": "correct",
            kc_col_name: "skill_id",
        }
    )

    # Create item from problem and step
    df["item_id"] = df["Problem Name"] + ":" + df["Step Name"]

    df = add_timestamp(df, "First Transaction Time")
    df = general_cleaning(df, min_interactions_per_user)
    df = remove_nan_skill(remove_nan_skills, df)

    # Extract KCs
    kc_list = []
    for kc_str in df["skill_id"].unique():
        for kc in kc_str.split("~~"):
            kc_list.append(kc)
    kc_set = set(kc_list)
    kc2idx = {kc: i for i, kc in enumerate(kc_set)}

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(kc_set)))
    for item_id, kc_str in df[["item_id", "skill_id"]].values:
        for kc in kc_str.split("~~"):
            Q_mat[item_id, kc2idx[kc]] = 1

    # Get unique skill id from combination of all skill ids
    df["skill_id"] = np.unique(Q_mat, axis=0, return_inverse=True)[1][df["item_id"]]

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    df.reset_index(inplace=True, drop=True)

    # Save data
    save_data(df, data_path, Q_mat)


def prepare_lalilo(min_interactions_per_user):
    """Preprocess Lalilo dataset.

    Arguments:
        min_interactions_per_user (int): minimum number of interactions per student

    Outputs:
        df (pandas DataFrame): preprocessed Lalilo dataset with user_id, item_id,
            timestamp and correct features
        Q_mat (item-skill relationships sparse array): corresponding q-matrix
    """
    data_path = os.path.join("data", "lalilo")
    df = pd.read_csv(
        os.path.join(data_path, "all_traces_from_2018-08-01_to_2019-04-01.csv")
    )

    def add_exercise_code_level_lesson(df):
        dataset = df.copy()
        dataset["exercise_code_level_lesson"] = (
            dataset["exercise_code"].map(str)
            + "_"
            + dataset["level"].map(str)
            + "_lesson_"
            + dataset["lesson_id"].map(str)
        )
        return dataset

    df = add_exercise_code_level_lesson(df)
    df = df.rename(
        columns={
            "student_id": "user_id",
            "created_at": "timestamp",
            "exercise_code": "skill_id",
            "exercise_code_level_lesson": "item_id",
            "correctness": "correct",
        }
    )
    df = add_timestamp(df, "timestamp")
    df = general_cleaning(df, min_interactions_per_user)

    # Maybe we want to store the correspondence with the original dataset somewhere
    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
    df["skill_id"] = np.unique(df["skill_id"], return_inverse=True)[1]

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    df = df[["user_id", "item_id", "skill_id", "timestamp", "correct"]]
    df.reset_index(inplace=True, drop=True)

    # Save data
    # save_data(df, data_path, Q_mat)


def general_cleaning(df, min_interactions_per_user):
    t1_start = process_time()
    # Remove continuous outcomes
    df = df.copy()
    df = df[df["correct"].isin([0, 1])]
    df["correct"] = df["correct"].astype(np.int32)
    # Drop duplicates
    df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], inplace=True)
    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)
    t1_stop = process_time()
    print("Elapsed time during general_cleaning in seconds:", t1_stop - t1_start)
    return df


def add_timestamp(df, column_name):
    t1_start = process_time()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[column_name])
    # df.dropna(subset=["timestamp"], inplace=True)
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df["timestamp"] = (
        df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
    )
    df.sort_values(by="timestamp", inplace=True)
    t1_stop = process_time()
    print("Elapsed time during add_timestamp in seconds:", t1_stop - t1_start)
    return df


def save_data(df, data_path, Q_mat):
    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), sparse.csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path, "preprocessed_data.csv"), sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets.")
    parser.add_argument("--dataset", type=str, default="assistments12")
    parser.add_argument("--min_interactions", type=int, default=10)
    parser.add_argument("--remove_nan_skills", type=bool, default=True)
    args = parser.parse_args()

    if args.dataset in [
        "assistments09",
        "assistments12",
        "assistments15",
        "assistments17",
    ]:
        prepare_assistments(
            data_name=args.dataset,
            min_interactions_per_user=args.min_interactions,
            remove_nan_skills=args.remove_nan_skills,
        )
    elif args.dataset == "bridge_algebra06":
        prepare_kddcup10(
            data_name="bridge_algebra06",
            min_interactions_per_user=args.min_interactions,
            kc_col_name="KC(SubSkills)",
            remove_nan_skills=args.remove_nan_skills,
        )
    elif args.dataset == "algebra05":
        prepare_kddcup10(
            data_name="algebra05",
            min_interactions_per_user=args.min_interactions,
            kc_col_name="KC(Default)",
            remove_nan_skills=args.remove_nan_skills,
        )
    elif args.dataset == "lalilo":
        prepare_lalilo(min_interactions_per_user=args.min_interactions)
