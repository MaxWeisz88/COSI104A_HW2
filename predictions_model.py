from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import scipy as sci
import numpy as np
import pandas as pd
import seaborn as sea
import time

train_data = pd.read_csv("HW2_training.csv")
# test_data = pd.read_csv("HW2_test_input.csv") 

def add_balance_feats(df):
    df = df.copy()

    df["orig_balance_error"] = (df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"])
    df["dest_balance_error"] = (df["oldbalanceDest"] + df["amount"] -df["newbalanceDest"])

    scale = df["amount"].clip(lower=1)
    df["orig_balance_error_scaled"] = df["orig_balance_error"] / scale
    df["dest_balance_error_scaled"] = df["dest_balance_error"] / scale

    df["orig_balance_was_zero"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["orig_balance_is_zero"] = (df["newbalanceOrig"] == 0).astype(int)
    df["dest_balance_was_zero"] = (df["oldbalanceDest"] == 0).astype(int)
    df["dest_balance_unchanged"] = (df["oldbalanceDest"] == df["newbalanceDest"]).astype(int)

    return df

train_data = add_balance_feats(train_data)
# print(train_data.head())
categorical_cols = train_data.select_dtypes(include="str").columns
numerical_cols = train_data.select_dtypes(exclude="str").columns
keep_cols = ["amount", "oldbalanceOrg", "newbalanceOrig"]
# train_data.drop(columns=["isFraud", "CASH_IN"])
# X = train_data[keep_cols]
X = train_data.drop(columns=["isFraud"])
y = train_data["isFraud"]

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
#                     stratify=y, random_state=83)

# X_fit, X_thresh, y_fit, y_thresh = train_test_split(X_train, y_train, 
#                     test_size=0.25, stratify=y_train, random_state=83)
# for column in X:
#     plt.figure()
#     for label, group in train_data.groupby("isFraud"):
#         plt.hist(group[column], bins=40, alpha=0.5, density=True,
#                  label=f"isFraud = {label}")
#     plt.title(f"{column} by fraud label")
#     plt.xlabel(f"log(1 + {column})")
#     plt.ylabel("Proportion")
#     plt.legend()
#     plt.show()

# fraud_rate = train_data.groupby("type")["isFraud"].mean()
# fraud_rate.plot(kind="bar")
# plt.title("Fraud rate by transaction type")
# plt.ylabel("Fraction marked as fraud")
# plt.show()



# print(numerical_cols)
# print(categorical_cols)

preprocessor = ColumnTransformer(transformers=[
    ("num", "passthrough", X.select_dtypes(include=["number"]).columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     X.select_dtypes(include=["str"]).columns)
    ])

# for col in ["amount", "oldbalanceOrg", "newbalanceOrig"]:
#     plt.figure()
#     train_data.groupby("isFraud")[col].plot(kind="hist", alpha=0.7, bins=30, legend=True)
#     plt.title(col)
#     plt.xlabel(col)
#     plt.show()

# sea.scatterplot(data=train_data, x="amount", y="oldbalanceOrg",
#                 hue="isFraud", alpha=0.6)
# plt.title("amount vs oldbalanceOrg")
# plt.show()

cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state= 83)
outer_f1_scores = []

# param_grid = {
#     "classifier__max_depth": [4, 8, 12, 20, None],
#     "classifier__min_samples_split": [2, 5, 10, 20, 50],
#     "classifier__min_samples_leaf": [1, 2, 5, 10, 20],
#     "classifier__max_features": [None, "sqrt", 0.7],
#     "classifier__ccp_alpha": [0.0, 1e-5, 1e-4, 1e-3],
#     "classifier__class_weight": [None, "balanced"],
# }

param_grid = {
    "classifier__max_depth": [10, 20, None],
        # "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__ccp_alpha": [0.00001]}
    #     ,"classifier__criterion": ["gini", "entropy", "log_loss"]
    #    }
# treeClassifier = tree.DecisionTreeClassifier(max_depth=depth, random_state=83, 
        # criterion="log_loss", min_samples_leaf=4, class_weight=None)
tree_classifier = tree.DecisionTreeClassifier(criterion="gini", ccp_alpha=1e-5,
                                    min_samples_split=2, random_state=83, class_weight=None)
model = Pipeline(steps=[("preprocess", preprocessor), ("classifier", tree_classifier)])
thresholds = np.linspace(0, 1, 1001)

for fold, (train_idx, val_idx) in enumerate(cv_outer.split(X, y), start=1):
    print(f"Starting out fold {fold}/5", flush=True)
    fold_start = time.perf_counter()

    X_train_out = X.iloc[train_idx]
    y_train_out = y.iloc[train_idx]
    X_val_out = X.iloc[val_idx]
    y_val_out = y.iloc[val_idx]

    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=83)
    # search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=30, scoring="f1",
    #                         cv=cv_inner, n_jobs=-1, random_state=83)
    search = GridSearchCV(model, param_grid, scoring="f1", cv=cv_inner, n_jobs=-1, verbose=2)
    search.fit(X_train_out, y_train_out)
    print(f"Grid search finished in {time.perf_counter()}", flush=True)

    predict_start = time.perf_counter()
    out_fold_probs = cross_val_predict(search.best_estimator_, X_train_out, 
                            y_train_out, cv=cv_inner, method="predict_proba", n_jobs=-1, verbose=2)[:, 1]
    print(f"Cross-validation predictions finished in "f"{time.perf_counter() - predict_start:.1f}s", flush=True)
    threshold_f1s = [f1_score(y_train_out, out_fold_probs >= threshold) 
                     for threshold in thresholds]
    best_thresh = thresholds[np.argmax(threshold_f1s)]

    val_probs = search.best_estimator_.predict_proba(X_val_out)[:, 1]
    val_preds = val_probs >= best_thresh
    fold_f1 = f1_score(y_val_out, val_preds)
    outer_f1_scores.append(fold_f1)

    print("Best parameters:", search.best_params_)
    print(f"Fold {fold}: threshold={best_thresh:.4f}, F1={fold_f1:.6f}")

print("Nested CV mean F1:", np.mean(outer_f1_scores))
print("Nested CV F1 standard deviation", np.std(outer_f1_scores))

cv_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=83)

final_search = GridSearchCV(model, param_grid, scoring="f1", cv=cv_final, n_jobs=-1, verbose=2)
final_search.fit(X, y)
print("Final best parameters:", final_search.best_params_)

oof_probs = cross_val_predict(final_search.best_estimator_, X, y, cv=cv_final, method="predict_proba"
                              , n_jobs=-1, verbose=2)[:, 1]

threshold_f1s = [f1_score(y, oof_probs >= threshold) for threshold in thresholds]
best_threshold = thresholds[np.argmax(threshold_f1s)]
print("Selected threshold:", best_threshold)

test_data = add_balance_feats(pd.read_csv("HW2_test_input.csv"))
X_test = test_data[X.columns]

test_probs = final_search.best_estimator_.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

pd.DataFrame({"isFraud": test_preds}).to_csv("HW2_predictions.csv", index=False)
print("Saved predictions to HW2_predictions.csv")


# search.fit(X_fit, y_fit)

# thresh_probs = search.best_estimator_.predict_proba(X_thresh)[:, 1]

# f1_scores = [f1_score(y_thresh, thresh_probs >= t) for t in thresholds]

# best_index = np.argmax(f1_scores)
# best_threshold = thresholds[best_index]

# eval_probs = search.best_estimator_.predict_proba(X_val)[:, 1]
# eval_preds = (eval_probs >= best_threshold).astype(int)

# print("Final evaluation F1:", f1_score(y_val, eval_preds))

# print("Best threshold:", best_thresh)
# print("Threshold-tuning F1:", f1_scores[best_index])

# for depth in [5, 7, 10, 15, 20]:
   
# model = Pipeline(steps=[("preprocess", preprocessor),
        # ("classifier", treeClassifier)])
    
# scores = cross_validate(model, X, y, cv=cv, scoring=["f1", "precision", "recall"], return_train_score=True)
# print("depth =", depth, "\nmean f1 =", scores["test_f1"].mean(), "\nmean precision =", 
#         scores["test_precision"].mean(), "\nmean recall = ", scores["test_recall"].mean(),
#             "\nfold f1 scores =", scores["test_f1"])

# treeClassifier = tree.DecisionTreeClassifier(max_depth=10, random_state=83,
#         criterion="gini", min_samples_leaf=5, class_weight="balanced")

# model = Pipeline(steps=[("preprocess", preprocessor), ("classifier", treeClassifier)])
# model.fit(X, y)

# importances = model.named_steps["classifier"].feature_importances_
# feature_names = model.named_steps["preprocess"].get_feature_names_out()

# importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}
#                              ).sort_values("importance", ascending=False)

# corr = train_data.corr(numeric_only=True)
# print(importance_df.head(10))
# print(corr["isFraud"].sort_values(ascending=False).head(10))


# print("Fold F1 scores:", scores["test_score"])
# print("Average F1:", scores["test_score"].mean())
