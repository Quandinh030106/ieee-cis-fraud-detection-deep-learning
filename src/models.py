import torch
import torch.nn as nn
from typing import Dict, List, Optional


class TabularEntityEmbeddingNN(nn.Module):
    """
    Deep Tabular Architecture with learned Entity Embeddings for categorical attributes
    and Batch-Normalized continuous feature streams.
    
    Mathematical Formulation:
        z_cat = Concat([E_i(x_i) for i in 1..49])
        z_num = BatchNorm(x_num)
        h_0   = Concat([z_cat, z_num])
        h_l   = Dropout(ReLU(BatchNorm(Linear(h_{l-1})))) + SkipConnection
        y_hat = Linear(h_L)  (Raw logit output)
    """
    def __init__(
        self,
        embedding_info: Dict[str, Dict[str, int]],
        cat_cols: List[str],
        num_features: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.25,
        use_residual: bool = True
    ):
        super().__init__()
        self.cat_cols = cat_cols
        self.use_residual = use_residual

        # 1. Dedicated Entity Embedding layers
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        for col in cat_cols:
            info = embedding_info[col]
            num_classes = info['num_classes']
            emb_dim = info['embedding_dim']
            emb_layer = nn.Embedding(num_classes, emb_dim)
            # Xavier normal initialization for embedding weights
            nn.init.xavier_normal_(emb_layer.weight)
            self.embeddings.append(emb_layer)
            total_emb_dim += emb_dim

        # 2. Continuous feature normalization
        self.num_bn = nn.BatchNorm1d(num_features) if num_features > 0 else nn.Identity()

        # 3. Backbone MLP with Residual skip-connections
        total_input_dim = total_emb_dim + num_features
        layers = nn.ModuleList()
        prev_dim = total_input_dim

        for hidden_dim in hidden_dims:
            block = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate)
            )
            layers.append(block)
            prev_dim = hidden_dim

        self.mlp_layers = layers

        # Optional linear projection for residual skip connection if dimensions mismatch
        if use_residual:
            self.skip_projections = nn.ModuleList()
            prev_dim = total_input_dim
            for hidden_dim in hidden_dims:
                if prev_dim != hidden_dim:
                    self.skip_projections.append(nn.Linear(prev_dim, hidden_dim, bias=False))
                else:
                    self.skip_projections.append(nn.Identity())
                prev_dim = hidden_dim

        # 4. Final classification head (unscaled logit output)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor) -> torch.Tensor:
        # cat_x: (B, num_cats), num_x: (B, num_features)
        emb_outputs = []
        for i, emb_layer in enumerate(self.embeddings):
            # Extract categorical column index
            col_tokens = cat_x[:, i]
            emb_outputs.append(emb_layer(col_tokens))

        z_cat = torch.cat(emb_outputs, dim=-1)
        z_num = self.num_bn(num_x)

        h = torch.cat([z_cat, z_num], dim=-1)

        # Forward through MLP layers with optional residual connections
        for idx, layer in enumerate(self.mlp_layers):
            if self.use_residual:
                skip = self.skip_projections[idx](h)
                h = layer(h) + skip
            else:
                h = layer(h)

        logits = self.head(h)
        return logits


class BaselineLinearClassifier(nn.Module):
    """
    Linear logistic regression baseline (for ablation and sanity check).
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
