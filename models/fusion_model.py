import torch
import torch.nn as nn
from models.selayer import SELayer

class DenseNetBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            SELayer(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            SELayer(64),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.block(x)
        return x.view(x.size(0), -1)

class AttentionFusion(nn.Module):
    def __init__(self, cnn_dim, evo_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(cnn_dim, cnn_dim)
        self.key_proj = nn.Linear(evo_dim, cnn_dim)
        self.value_proj = nn.Linear(evo_dim, cnn_dim)
        self.mha = nn.MultiheadAttention(embed_dim=cnn_dim, num_heads=num_heads, batch_first=True)

    def forward(self, cnn_feat, evo_feat):
        q = self.query_proj(cnn_feat).unsqueeze(1)
        k = self.key_proj(evo_feat)
        v = self.value_proj(evo_feat)
        fused, _ = self.mha(q, k, v)
        return fused.squeeze(1)

class FusionModel(nn.Module):
    def __init__(self, evo_dim=4096, struct_dim=802):
        super().__init__()
        self.cnn_branch = DenseNetBranch()
        self.evo_proj = nn.Linear(evo_dim, 64)
        self.struct_proj = nn.Linear(struct_dim, 128)
        self.attn_fuse = AttentionFusion(cnn_dim=64, evo_dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(64 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, cnn_x, evo_x, struct_x, return_feat=False):
        cnn_feat = self.cnn_branch(cnn_x)
        evo_feat = self.evo_proj(evo_x)
        struct_feat = self.struct_proj(struct_x)
        fused_feat = self.attn_fuse(cnn_feat, evo_feat)
        combined = torch.cat([fused_feat, struct_feat], dim=1)
        out = self.classifier(combined)
        if return_feat:
            return out, combined
        return out