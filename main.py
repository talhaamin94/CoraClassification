# import numpy as np
# from sklearn.model_selection import StratifiedKFold, GridSearchCV
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LogisticRegression
# import pandas as pd
# from load_data import load_cora_data

# # Load data
# features, labels, paper_ids, _ = load_cora_data()

# # Initialize cross-validation
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# all_preds = np.zeros_like(labels)
# all_ids = np.array(paper_ids)

# # Grid search parameters
# param_grid = {
#     "C": [0.001,0.005,0.01, 0.02, 0.03, 0.05],
#     "penalty": ["l2"],
#     "solver": ["lbfgs"],
#     "max_iter": [1000],
#     "class_weight": ["balanced"] 
# }

# for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
#     # Feature Scaling
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(features[train_idx])
#     X_test = scaler.transform(features[test_idx])
#     y_train = labels[train_idx]
#     y_test = labels[test_idx]

#     # Grid search with class balancing
#     grid = GridSearchCV(LogisticRegression(), param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
#     grid.fit(X_train, y_train)

#     best_model = grid.best_estimator_
#     preds = best_model.predict(X_test)
#     all_preds[test_idx] = preds

#     acc = accuracy_score(y_test, preds)
#     f1 = f1_score(y_test, preds, average="macro")
#     print(f"Fold {fold+1} Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Best C: {grid.best_params_['C']}")

# # Evaluate overall performance
# overall_acc = accuracy_score(labels, all_preds)
# overall_f1 = f1_score(labels, all_preds, average="macro")
# print(f"\nOverall Accuracy: {overall_acc:.4f}")
# print(f"Overall F1 Score (Macro): {overall_f1:.4f}")

# # Output predictions
# content = pd.read_csv("cora/cora.content", sep="\t", header=None)
# original_labels = content.iloc[:, -1].values
# label_encoder = LabelEncoder().fit(original_labels)
# decoded_preds = label_encoder.inverse_transform(all_preds)

# df_out = pd.DataFrame({"paper_id": paper_ids, "class_label": decoded_preds})
# df_out.to_csv("predictions.tsv", sep="\t", index=False, header=False)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from load_data import load_cora_data  # Make sure this returns `data, label_encoder`

# Define GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Load data
data, label_encoder = load_cora_data()
features = data.x.cpu().numpy()
labels = data.y.cpu().numpy()
edge_index = data.edge_index
paper_ids = pd.read_csv("cora/cora.content", sep="\t", header=None).iloc[:, 0].values

# Cross-validation setup
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_preds = np.zeros_like(labels)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
    # Feature scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    x = torch.tensor(features_scaled, dtype=torch.float)

    # New data object per fold
    fold_data = Data(x=x, edge_index=edge_index, y=torch.tensor(labels, dtype=torch.long))

    # Define masks
    fold_data.train_mask = torch.zeros(len(labels), dtype=torch.bool)
    fold_data.test_mask = torch.zeros(len(labels), dtype=torch.bool)
    fold_data.train_mask[train_idx] = True
    fold_data.test_mask[test_idx] = True

    fold_data = fold_data.to(device)

    model = GCN(in_channels=fold_data.num_node_features, hidden_channels=16, out_channels=len(np.unique(labels))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(fold_data.x, fold_data.edge_index)
        loss = F.cross_entropy(out[fold_data.train_mask], fold_data.y[fold_data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(mask):
        model.eval()
        with torch.no_grad():
            out = model(fold_data.x, fold_data.edge_index)
            pred = out[mask].argmax(dim=1)
            y_true = fold_data.y[mask]
            acc = accuracy_score(y_true.cpu(), pred.cpu())
            f1 = f1_score(y_true.cpu(), pred.cpu(), average="macro")
            return acc, f1, pred.cpu().numpy()

    # Training
    for epoch in range(1, 201):
        loss = train()
        if epoch % 20 == 0 or epoch == 1:
            acc, f1, _ = evaluate(fold_data.test_mask)
            print(f"Fold {fold+1} | Epoch {epoch} | Loss: {loss:.4f} | Test Acc: {acc:.4f} | Test F1: {f1:.4f}")

    # Final prediction for this fold
    _, _, preds = evaluate(fold_data.test_mask)
    all_preds[test_idx] = preds

    acc, f1, _ = evaluate(fold_data.test_mask)
    print(f"\nFold {fold+1} Final -- Accuracy: {acc:.4f}, F1 Score: {f1:.4f}\n")

# Final overall performance
overall_acc = accuracy_score(labels, all_preds)
overall_f1 = f1_score(labels, all_preds, average="macro")
print(f"\n Overall Accuracy: {overall_acc:.4f}")
print(f" Overall Macro F1 Score: {overall_f1:.4f}")

# Save final predictions
decoded_preds = label_encoder.inverse_transform(all_preds)
df_out = pd.DataFrame({
    "paper_id": paper_ids,
    "class_label": decoded_preds
})
df_out.to_csv("predictions.tsv", sep="\t", index=False, header=False)
