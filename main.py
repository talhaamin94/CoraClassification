import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
from load_data import load_cora_data

# Load data
features, labels, paper_ids, _ = load_cora_data()

# Initialize cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_preds = np.zeros_like(labels)
all_ids = np.array(paper_ids)

# Grid search parameters
param_grid = {
    "C": [0.001,0.005,0.01, 0.02, 0.03, 0.05],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
    "max_iter": [1000],
    "class_weight": ["balanced"] 
}

for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train_idx])
    X_test = scaler.transform(features[test_idx])
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    # Grid search with class balancing
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    all_preds[test_idx] = preds

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"Fold {fold+1} Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Best C: {grid.best_params_['C']}")

# Evaluate overall performance
overall_acc = accuracy_score(labels, all_preds)
overall_f1 = f1_score(labels, all_preds, average="macro")
print(f"\nOverall Accuracy: {overall_acc:.4f}")
print(f"Overall F1 Score (Macro): {overall_f1:.4f}")

# Output predictions
content = pd.read_csv("cora/cora.content", sep="\t", header=None)
original_labels = content.iloc[:, -1].values
label_encoder = LabelEncoder().fit(original_labels)
decoded_preds = label_encoder.inverse_transform(all_preds)

df_out = pd.DataFrame({"paper_id": paper_ids, "class_label": decoded_preds})
df_out.to_csv("predictions.tsv", sep="\t", index=False, header=False)
