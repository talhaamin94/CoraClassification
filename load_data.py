# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# def load_cora_data(path="cora/cora.content", cite_path="cora/cora.cites"):
#     # Load content
#     content = pd.read_csv(path, sep="\t", header=None)
#     features = content.iloc[:, 1:-1].values
#     labels = LabelEncoder().fit_transform(content.iloc[:, -1])
#     paper_ids = content.iloc[:, 0].values

#     # Map paper ID to index
#     paper_id_to_idx = {id_: idx for idx, id_ in enumerate(paper_ids)}

#     return features, labels, paper_ids, paper_id_to_idx


import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

def load_cora_data(path="cora/cora.content", cite_path="cora/cora.cites"):
    # Load node content
    content = pd.read_csv(path, sep="\t", header=None)
    features = content.iloc[:, 1:-1].values
    paper_ids = content.iloc[:, 0].values
    labels_raw = content.iloc[:, -1].values

    # print(paper_ids)
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_raw)

    # Map paper IDs to indices
    paper_id_to_idx = {str(id_): idx for idx, id_ in enumerate(paper_ids)}

    edge_list = []
    missing_src = 0
    missing_dst = 0
    # print(paper_id_to_idx)
    with open(cite_path, "r") as f:
        for line in f:
            src, dst = line.strip().split()
            src = src.strip()
            dst = dst.strip()
            
            if src in paper_id_to_idx and dst in paper_id_to_idx:
                edge_list.append([paper_id_to_idx[src], paper_id_to_idx[dst]])
                edge_list.append([paper_id_to_idx[dst], paper_id_to_idx[src]])  # undirected
            else:
                if src not in paper_id_to_idx:
                    missing_src += 1
                if dst not in paper_id_to_idx:
                    missing_dst += 1

    # print(f"Loaded {len(edge_list)} citation edges.")
    # print(f"Missing source papers: {missing_src}")
    # print(f"Missing destination papers: {missing_dst}")

    # print(f"Loaded {len(edge_list)//2} citation edges.")
    # print(f"Missing source papers: {missing_src}")
    # print(f"Missing destination papers: {missing_dst}")


    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # Build random masks (standard: 140 train, 500 val, 1000 test)
    num_nodes = x.size(0)
    indices = np.arange(num_nodes)
    np.random.seed(42)
    np.random.shuffle(indices)

    # train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # train_mask[indices[:140]] = True
    # val_mask[indices[140:640]] = True
    # test_mask[indices[640:1640]] = True

    # Create PyG Data object
    data = Data(x=x, y=y, edge_index=edge_index)
                # train_mask=train_mask,
                # val_mask=val_mask,
                # test_mask=test_mask)

    return data, label_encoder
