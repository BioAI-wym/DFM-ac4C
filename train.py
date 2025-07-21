import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.fusion_model import FusionModel
from data.preprocess import encode_sequence

class SequenceDataset(Dataset):
    def __init__(self, seqs, labels, evo_feats, struct_feats):
        self.x_seq = np.stack([encode_sequence(s) for s in seqs])[:, None, :, :]
        self.evo_feats = evo_feats.float()
        self.struct = struct_feats.float()
        self.y = labels.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x_seq[idx], dtype=torch.float32),
            self.evo_feats[idx],
            self.struct[idx],
            torch.tensor(self.y[idx], dtype=torch.long)
        )

def load_fasta(path, pos_limit):
    seqs, labels = [], []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    for i in range(0, len(lines), 2):
        seqs.append(lines[i+1].upper().replace('T', 'U'))
        labels.append(1 if i//2 < pos_limit else 0)
    return seqs, np.array(labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_seqs, train_labels = load_fasta("data/iRNA-ac4c-trainset.txt", 2206)
train_evo = torch.load("model/28-train_features_irna_ac4c.pt")['X']
train_struct = pd.concat([
    pd.read_csv("fea/DNAShape/train_pos_final_feature_matrix.csv", header=None),
    pd.read_csv("fea/DNAShape/train_neg_final_feature_matrix.csv", header=None)
], ignore_index=True).fillna(0)
train_struct = torch.tensor(train_struct.values, dtype=torch.float32)

train_ds = SequenceDataset(train_seqs, train_labels, train_evo, train_struct)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

model = FusionModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(1, 101):
    model.train()
    for x_seq, evo, struct, y in train_loader:
        x_seq, evo, struct, y = x_seq.to(device), evo.to(device), struct.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x_seq, evo, struct)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch:03d} done")

torch.save(model.state_dict(), "trained_model.pt")