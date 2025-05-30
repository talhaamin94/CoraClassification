# GCN Training and Evaluation on Cora Dataset

We updated the script so now it trains a simple two-layer Graph Convolutional Network (GCN) on the Cora citation dataset using **stratified 10-fold cross-validation**. It reports accuracy and macro F1 scores for each fold and overall performance.
**(The previous script used logistic regression, the code is commented out but you can enable it and run)**

---

## Overview

1. **Data Loading**

   - The dataset is loaded via a custom `load_cora_data()` function which returns a PyTorch Geometric `Data` object with:
     - Node features (`x`)
     - Edge indices (`edge_index`)
     - Node labels (`y`)
   - Paper IDs are loaded separately from the original Cora `.content` file for later association with predictions.
   - A label encoder is used to map numeric labels back to original class names.

2. **Model Definition**
   - A 2-layer GCN with ReLU activation and dropout:
     - Input → GCNConv (hidden 16) → ReLU → Dropout → GCNConv (output classes)
3. **Cross-Validation Setup**

   - Stratified 10-fold cross-validation ensures class balance across folds.
   - In each fold:
     - Features are scaled with `StandardScaler`.
     - Train/test masks are created to select nodes for training and testing.

4. **Training and Evaluation**

   - The GCN is trained for 200 epochs per fold using Adam optimizer.
   - Cross-entropy loss is computed only on train nodes.
   - After training, model performance is evaluated on test nodes using accuracy and macro F1 score.
   - Predictions are collected across folds for final performance assessment.

5. **Results Saving**
   - Final predictions for all nodes are inverse transformed to class labels.
   - Results are saved to `predictions.tsv` with columns: `paper_id`, `class_label`.

---

## Important Notes on Train/Validation/Test Splits

- This script uses **10-fold cross-validation only**, meaning:

  - In each fold, 90% of nodes are training data and 10% are test data.
  - There is **no separate validation set** for hyperparameter tuning or early stopping.
  - All nodes are used in training/testing across the folds.

- This differs from the **fixed train/val/test splits** often used in PyG benchmarks for Cora, which typically have:

  - 140 training nodes
  - 500 validation nodes
  - 1000 test nodes

- If you want a fixed validation set or early stopping, you may consider:
  - Splitting training folds further into train and validation sets.
  - Or use the fixed train/val/test masks without cross-validation.

---

## Usage

- Ensure you have the necessary dependencies installed:

```bash
pip install torch torch_geometric scikit-learn pandas numpy
```
