"""
PatchTST: Patch Time Series Transformer (Nie et al., 2023).

Key innovation: segment time series into patches (like ViT for images),
then apply Transformer encoder. Much faster than per-point attention
and better at capturing long-range dependencies.

Paper: https://arxiv.org/abs/2211.14730
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import math


# ── Reversible Instance Normalization ────────────────────────────

class RevIN(nn.Module):
    """RevIN: Reversible Instance Normalization (Kim et al., ICLR 2022).

    PatchTST 论文的关键组件之一：每个样本自己做归一化，预测后再反归一化。
    电力负荷有明显日周期和水平漂移，不做 instance norm 时长程预测会掉点。
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        self._mean = None
        self._stdev = None

    def forward(self, x, mode: str):
        if mode == "norm":
            # x: (batch, seq_len, n_vars)
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._stdev = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.weight + self.bias
            return x

        if mode == "denorm":
            # x: (batch, pred_len) —— 目标只有负荷这一路，用第 0 路的统计量还原
            mean = self._mean[:, 0, 0].unsqueeze(-1)
            stdev = self._stdev[:, 0, 0].unsqueeze(-1)
            if self.affine:
                x = (x - self.bias[0]) / self.weight[0]
            return x * stdev + mean

        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")


# ── Positional Encoding ─────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


# ── Patch Embedding ──────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Split a univariate or multivariate series into patches and project."""

    def __init__(self, d_model: int, patch_len: int, stride: int, n_vars: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_vars = n_vars

        # Linear projection per variable
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, n_vars)
        batch, seq_len, n_vars = x.shape
        patches = []

        for v in range(n_vars):
            # Slide over the variable
            var_patches = x[:, :, v].unfold(dimension=1, size=self.patch_len, step=self.stride)
            patches.append(var_patches)
        # (batch, n_vars, n_patches, patch_len)
        patches = torch.stack(patches, dim=1)

        # Reshape to (batch * n_vars, n_patches, patch_len)
        patches = patches.reshape(batch * n_vars, patches.shape[2], self.patch_len)

        # Project each patch to d_model
        embedded = self.projection(patches)  # (batch * n_vars, n_patches, d_model)
        return self.dropout(embedded), patches.shape[1]


# ── PatchTST Encoder ─────────────────────────────────────────────

class PatchTSTEncoder(nn.Module):
    def __init__(
        self,
        n_vars: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(d_model, patch_len, stride, n_vars, dropout)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout_final = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, n_vars)
        batch, seq_len, n_vars = x.shape

        # Patch embedding
        embedded, n_patches = self.patch_embed(x)
        # embedded: (batch * n_vars, n_patches, d_model)

        # Positional encoding
        embedded = self.pos_encoding(embedded)

        # Transformer
        encoded = self.transformer(embedded)
        # encoded: (batch * n_vars, n_patches, d_model)

        # Flatten and reshape back
        encoded = self.flatten(encoded)  # (batch * n_vars, n_patches * d_model)
        encoded = encoded.reshape(batch, n_vars, -1)  # (batch, n_vars, n_patches * d_model)
        encoded = encoded.reshape(batch, -1)  # (batch, n_vars * n_patches * d_model)

        return self.dropout_final(encoded)


# ── Full PatchTST Model ──────────────────────────────────────────

class PatchTST(nn.Module):
    def __init__(
        self,
        n_vars: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        use_revin: bool = True,
    ):
        super().__init__()
        self.use_revin = use_revin
        self.revin = RevIN(n_vars) if use_revin else None

        self.encoder = PatchTSTEncoder(
            n_vars=n_vars,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            patch_len=patch_len,
            stride=stride,
            dropout=dropout,
        )

        # Compute encoder output dim
        n_patches = (seq_len - patch_len) // stride + 1
        self.encoder_dim = n_vars * n_patches * d_model

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(self.encoder_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, pred_len),
        )

    def forward(self, x):
        if self.revin is not None:
            x = self.revin(x, "norm")
        out = self.head(self.encoder(x))
        if self.revin is not None:
            out = self.revin(out, "denorm")
        return out


# ── Trainer ──────────────────────────────────────────────────────

class PatchTSTPredictor:
    def __init__(self, seq_len=96, pred_len=24, n_vars=12, use_revin=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.use_revin = use_revin
        self.model = None
        self.config = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

        print(f"Using device: {self.device}")

    def build_model(
        self,
        d_model=128,
        n_heads=8,
        n_layers=3,
        patch_len=16,
        stride=8,
        dropout=0.1,
        learning_rate=0.001,
        use_revin=None,
    ):
        if use_revin is None:
            use_revin = self.use_revin
        self.use_revin = use_revin

        self.model = PatchTST(
            n_vars=self.n_vars,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            patch_len=patch_len,
            stride=stride,
            dropout=dropout,
            use_revin=use_revin,
        ).to(self.device)

        # 存档用配置（train() 里保存 checkpoint 时要用）
        self.config = {
            "seq_len": self.seq_len, "pred_len": self.pred_len, "n_vars": self.n_vars,
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "patch_len": patch_len, "stride": stride, "use_revin": use_revin,
        }

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"PatchTST built: {total_params:,} params ({trainable:,} trainable)")
        return self.model

    def train(
        self, X_train, y_train, X_val, y_val,
        epochs=100, batch_size=32, patience=15,
        save_path="models/patchtst_model.pth",
    ):
        if self.model is None:
            raise ValueError("Call build_model() first")

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=batch_size, shuffle=False
        )

        print(f"Training PatchTST: {len(X_train)} train, {len(X_val)} val")
        best_val_loss = float("inf")
        patience_counter = 0
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in range(epochs):
            # ── Train ──
            self.model.train()
            train_loss, train_mae = 0.0, 0.0
            for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - batch_y)).item()

            # ── Val ──
            self.model.eval()
            val_loss, val_mae = 0.0, 0.0
            with torch.no_grad():
                for batch_X, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(outputs - batch_y)).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["train_mae"].append(train_mae / len(train_loader))
            self.history["val_mae"].append(val_mae / len(val_loader))

            self.scheduler.step(avg_val_loss)

            print(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss={avg_train_loss:.6f} | Val Loss={avg_val_loss:.6f} | "
                f"LR={self.optimizer.param_groups[0]['lr']:.8f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                    "best_val_loss": best_val_loss,
                }, save_path)
                print(f"  ✅ Saved (Val Loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Training done. Best val loss: {best_val_loss:.6f}")
        return self.history

    def predict(self, X):
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            return self.model(X_t).cpu().numpy()

    def evaluate(self, X_test, y_test, preprocessor=None):
        predictions = self.predict(X_test)

        pred_orig = predictions.flatten()
        y_orig = y_test.flatten()

        if preprocessor is not None:
            # 目标和预测都还在 StandardScaler 的标准化空间（z-score）里，都要还原回 MW。
            # 注意 RevIN 的 denorm 只撤销它自己加上的 instance norm：输入原本就是 scaler
            # 标准化过的，所以 RevIN 的输出仍是 z-score，不是 MW —— 有 RevIN 时也必须再
            # 过一次 scaler 反变换，否则会拿 z-score 去和 MW 比，指标全是废的。
            y_orig = preprocessor.inverse_transform_load(y_orig)
            pred_orig = preprocessor.inverse_transform_load(pred_orig)

        rmse = np.sqrt(mean_squared_error(y_orig, pred_orig))
        mae = mean_absolute_error(y_orig, pred_orig)
        r2 = r2_score(y_orig, pred_orig)
        mape = np.mean(np.abs((y_orig - pred_orig) / y_orig)) * 100

        metrics = {"RMSE": rmse, "MAE": mae, "R²": r2, "MAPE": mape}

        print(f"PatchTST Results: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")
        return metrics, pred_orig, y_orig

    def plot_history(self):
        if not self.history["train_loss"]:
            return
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.title("PatchTST Loss")
        plt.legend(); plt.grid(alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_mae"], label="Train MAE")
        plt.plot(self.history["val_mae"], label="Val MAE")
        plt.title("PatchTST MAE")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("models/patchtst_training_history.png", dpi=150, bbox_inches="tight")
        plt.show()

    def load_model(self, filepath="models/patchtst_model.pth"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        cfg = checkpoint["config"]
        self.seq_len = cfg["seq_len"]
        self.pred_len = cfg["pred_len"]
        self.n_vars = cfg["n_vars"]
        self.use_revin = cfg.get("use_revin", True)

        self.model = PatchTST(
            n_vars=cfg["n_vars"],
            seq_len=cfg["seq_len"],
            pred_len=cfg["pred_len"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            patch_len=cfg["patch_len"],
            stride=cfg["stride"],
            use_revin=self.use_revin,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded PatchTST from {filepath}")
        return self.model
