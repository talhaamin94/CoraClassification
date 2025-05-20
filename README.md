# Cora Dataset Classification with Logistic Regression

This project performs node classification on the [Cora citation dataset](https://relational.fit.cvut.cz/dataset/CORA) using logistic regression and 10-fold stratified cross-validation. Feature scaling and hyperparameter tuning with `GridSearchCV` are applied for optimal performance.

---

## 🧠 Features

- Loads and preprocesses Cora dataset
- Uses `StandardScaler` for normalization
- Applies `LogisticRegression` with `GridSearchCV` to find best hyperparameters
- Evaluates performance with accuracy and macro F1 score
- Saves final predictions in `predictions.tsv`

---

## 🚀 Usage

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/cora-logistic-regression.git
cd cora-logistic-regression
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the classification script

```bash

python main.py

```
