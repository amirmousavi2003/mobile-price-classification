import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42
TEST_SIZE = 0.30

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = "mobile_price_range_data.csv"

df = pd.read_csv(DATA_PATH)

plt.figure(figsize=(6, 4))
sns.countplot(x="price_range", data=df)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
plt.close()

X = df.drop(columns=["price_range"])
y = df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate_model(model, X_tr, y_tr, X_te, y_te, model_name, params_str):
    model.fit(X_tr, y_tr)

    y_pred_train = model.predict(X_tr)
    y_pred_test = model.predict(X_te)

    train_acc = accuracy_score(y_tr, y_pred_train)
    test_acc = accuracy_score(y_te, y_pred_test)
    precision = precision_score(y_te, y_pred_test, average="macro", zero_division=0)
    recall = recall_score(y_te, y_pred_test, average="macro", zero_division=0)
    f1 = f1_score(y_te, y_pred_test, average="macro", zero_division=0)

    result = {
        "Model": model_name,
        "Params": params_str,
        "Train_Accuracy": round(train_acc, 4),
        "Test_Accuracy": round(test_acc, 4),
        "Overfitting_Gap": round(train_acc - test_acc, 4),
        "Precision_macro": round(precision, 4),
        "Recall_macro": round(recall, 4),
        "F1_macro": round(f1, 4),
    }
    return result, model, y_pred_test


def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


dt_results = []
depth_values = [2, 3, 4, 5, 7, 10, 15, None]
criteria = ["gini", "entropy"]

for criterion in criteria:
    for depth in depth_values:
        dt = DecisionTreeClassifier(
            max_depth=depth, criterion=criterion, random_state=RANDOM_STATE
        )
        params_str = f"max_depth={depth}, criterion={criterion}"
        result, fitted_model, y_pred = evaluate_model(
            dt, X_train, y_train, X_test, y_test, "Decision Tree", params_str
        )
        dt_results.append(result)

dt_results_df = pd.DataFrame(dt_results)
dt_results_df.to_csv(os.path.join(OUTPUT_DIR, "decision_tree_results.csv"), index=False)

gini_df = dt_results_df[dt_results_df["Params"].str.contains("gini")].copy()
gini_df["depth_num"] = gini_df["Params"].str.extract(r"max_depth=(\d+|None)")
gini_df["depth_num"] = gini_df["depth_num"].replace("None", "20").astype(int)
gini_df = gini_df.sort_values("depth_num")

plt.figure(figsize=(7, 5))
plt.plot(gini_df["depth_num"], gini_df["Train_Accuracy"], marker="o", label="Train Accuracy")
plt.plot(gini_df["depth_num"], gini_df["Test_Accuracy"], marker="s", label="Test Accuracy")
plt.xlabel("max_depth (20 = None)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dt_overfitting_plot.png"), dpi=150)
plt.close()

best_dt_row = dt_results_df.loc[dt_results_df["F1_macro"].idxmax()]

best_depth = None if "None" in best_dt_row["Params"] else int(
    best_dt_row["Params"].split("max_depth=")[1].split(",")[0]
)
best_criterion = best_dt_row["Params"].split("criterion=")[1]
best_dt_model = DecisionTreeClassifier(
    max_depth=best_depth, criterion=best_criterion, random_state=RANDOM_STATE
)
best_dt_model.fit(X_train, y_train)
y_pred_best_dt = best_dt_model.predict(X_test)

plot_confusion(y_test, y_pred_best_dt,
               f"Decision Tree ({best_dt_row['Params']})",
               "confusion_matrix_decision_tree.png")

plt.figure(figsize=(16, 8))
plot_tree(best_dt_model, max_depth=3, feature_names=X.columns, filled=True,
          class_names=[str(c) for c in sorted(y.unique())], fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "decision_tree_structure.png"), dpi=150)
plt.close()


knn_results = []
k_values = [1, 3, 5, 7, 9, 11, 15, 21, 25, 31, 41, 51, 61, 71, 81, 101]
distance_metrics = ["euclidean", "manhattan"]

for metric in distance_metrics:
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        params_str = f"K={k}, metric={metric}"
        result, fitted_model, y_pred = evaluate_model(
            knn, X_train_scaled, y_train, X_test_scaled, y_test, "KNN", params_str
        )
        knn_results.append(result)

knn_results_df = pd.DataFrame(knn_results)
knn_results_df.to_csv(os.path.join(OUTPUT_DIR, "knn_results.csv"), index=False)

plt.figure(figsize=(7, 5))
for metric in distance_metrics:
    sub = knn_results_df[knn_results_df["Params"].str.contains(f"metric={metric}")].copy()
    sub["k_num"] = sub["Params"].str.extract(r"K=(\d+)").astype(int)
    sub = sub.sort_values("k_num")
    plt.plot(sub["k_num"], sub["Test_Accuracy"], marker="o", label=f"{metric}")

plt.xlabel("K")
plt.ylabel("Test Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "knn_k_tuning_plot.png"), dpi=150)
plt.close()

best_knn_row = knn_results_df.loc[knn_results_df["F1_macro"].idxmax()]

best_k = int(best_knn_row["Params"].split("K=")[1].split(",")[0])
best_metric = best_knn_row["Params"].split("metric=")[1]
best_knn_model = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
best_knn_model.fit(X_train_scaled, y_train)
y_pred_best_knn = best_knn_model.predict(X_test_scaled)

plot_confusion(y_test, y_pred_best_knn,
               f"KNN ({best_knn_row['Params']})",
               "confusion_matrix_knn.png")

final_comparison = pd.DataFrame([
    {
        "Model": "Decision Tree (Best)",
        "Params": best_dt_row["Params"],
        "Test_Accuracy": best_dt_row["Test_Accuracy"],
        "Precision_macro": best_dt_row["Precision_macro"],
        "Recall_macro": best_dt_row["Recall_macro"],
        "F1_macro": best_dt_row["F1_macro"],
    },
    {
        "Model": "KNN (Best)",
        "Params": best_knn_row["Params"],
        "Test_Accuracy": best_knn_row["Test_Accuracy"],
        "Precision_macro": best_knn_row["Precision_macro"],
        "Recall_macro": best_knn_row["Recall_macro"],
        "F1_macro": best_knn_row["F1_macro"],
    },
])
final_comparison.to_csv(os.path.join(OUTPUT_DIR, "final_comparison.csv"), index=False)