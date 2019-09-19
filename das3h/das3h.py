import pandas as pd
from datetime import date

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from encode import encode_df
from clean import clean_df
from results_analysis import (
    get_coefs_in_dataframe,
    get_students_alphas,
    get_exercise_code_betas,
    get_exercise_gammas_of_one_exercise_code,
    get_available_exercise_codes,
)

csv_to_use = "shorter_training"
# put the dataset you want in a folder with the right path
dataset = pd.read_csv(f"data/lalilo_datasets/{csv_to_use}.csv")

# select end of the dataset to test if the model is running properly
last_n_traces = 500  
if last_n_traces:
    dataset = dataset[-last_n_traces:]

# clean and encode dataset
cleaned_dataset = clean_df(dataset)
encoded_dataset = encode_df(cleaned_dataset)

# X and y, X has to be sparse when training a huge dataset
X = encoded_dataset.drop(columns=["correctness", "timestamp"])
X_sparse_df = X.astype(pd.SparseDtype("float", 0.0))
X_sparse_array = X_sparse_df.sparse.to_coo()

y = encoded_dataset["correctness"]

model = LogisticRegression(solver="lbfgs", max_iter=800)
model.fit(X, y)

coefficients = get_coefs_in_dataframe(model, X)

# saving the coefficients, you will have to create a 'results' folder somewhere
save_coeffs = False
if save_coeffs:
    today = date.today().strftime("%Y-%m-%d")
    coefficients.to_csv(f"das3h/results/coefficients_of_{csv_to_use}_done_{today}.csv")

print("Printing students alphas")
print(get_students_alphas(coefficients))
print("")
print("Printing exercise_code betas")
print(get_exercise_code_betas(coefficients))

for exercise_code in get_available_exercise_codes(coefficients):
    print("")
    print(f"Printing coefs of {exercise_code} for all its levels and lessons")
    print(get_exercise_gammas_of_one_exercise_code(coefficients, exercise_code))

