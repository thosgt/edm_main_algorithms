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
dataset = pd.read_csv(f"data/lalilo_datasets/{csv_to_use}.csv")[:500]
cleaned_dataset = clean_df(dataset)
encoded_dataset = encode_df(cleaned_dataset)

X = encoded_dataset.drop(columns=["correctness", "timestamp"])
y = encoded_dataset["correctness"]

model = LogisticRegression(solver="lbfgs", max_iter=400)
model.fit(X, y)

today = date.today().strftime("%Y-%m-%d")
coefficients = get_coefs_in_dataframe(model, X)
coefficients.to_csv(f"results/coefficients_of_{csv_to_use}_{today}.csv")

print("Printing students alphas")
print(get_students_alphas(coefficients))
print("")
print("Printing exercise_code betas")
print(get_exercise_code_betas(coefficients))

for exercise_code in get_available_exercise_codes(coefficients):
    print("")
    print(f"Printing coefs of {exercise_code} for all its levels and lessons")
    print(get_exercise_gammas_of_one_exercise_code(coefficients, exercise_code))

