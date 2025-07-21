import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, matthews_corrcoef
)
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

test_seqs, test_labels = load_fasta("data/iRNA-ac4c-testset.txt", 552)
test_evo = torch.load("model/28-test_features_irna_ac4c.pt")['X']
test_struct = pd.concat([
    pd.read_csv("fea/DNAShape/test_pos_final_feature_matrix.csv", header=None),
    pd.read_csv("fea/DNAShape/test_neg_final_feature_matrix.csv", header=None)
], ignore_index=True).fillna(0)
test_struct = torch.tensor(test_struct.values, dtype=torch.float32)

test_ds = SequenceDataset(test_seqs, test_labels, test_evo, test_struct)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

model = FusionModel().to(device)
model.load_state_dict(torch.load("trained_model.pt"))
model.eval()

scores, labels = [], []
with torch.no_grad():
    for x_seq, evo, struct, y in test_loader:
        x_seq, evo, struct = x_seq.to(device), evo.to(device), struct.to(device)
        out = model(x_seq, evo, struct)
        prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        scores.extend(prob)
        labels.extend(y.numpy())

auc = roc_auc_score(labels, scores)
pred = (np.array(scores) >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
sen = tp / (tp + fn) * 100
spe = tn / (tn + fp) * 100
acc = accuracy_score(labels, pred) * 100
mcc = matthews_corrcoef(labels, pred) * 100

print(f"AUROC={auc:.2f}% | SEN={sen:.2f}% | SPE={spe:.2f}% | ACC={acc:.2f}% | MCC={mcc:.2f}%")