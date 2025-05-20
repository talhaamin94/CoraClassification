import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_cora_data(path="cora/cora.content", cite_path="cora/cora.cites"):
    # Load content
    content = pd.read_csv(path, sep="\t", header=None)
    features = content.iloc[:, 1:-1].values
    labels = LabelEncoder().fit_transform(content.iloc[:, -1])
    paper_ids = content.iloc[:, 0].values

    # Map paper ID to index
    paper_id_to_idx = {id_: idx for idx, id_ in enumerate(paper_ids)}

    return features, labels, paper_ids, paper_id_to_idx
