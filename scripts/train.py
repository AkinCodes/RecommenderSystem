import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.dlrm import DLRMModel
from scripts.preprocessing import fit_and_save_scaler, fit_and_save_encoders


class DLRMTrainer(pl.LightningModule):
    def __init__(self, num_features, embedding_sizes, mlp_layers, lr=1e-3):
        super().__init__()
        self.model = DLRMModel(num_features, embedding_sizes, mlp_layers)
        self.loss_fn = nn.BCELoss()
        self.lr = lr
        self.automatic_optimization = False

    def forward(self, continuous_features, categorical_features):
        return self.model(continuous_features, categorical_features)

    def training_step(self, batch, batch_idx):
        continuous, categorical, labels = batch
        labels = labels.float().view(-1, 1)
        predictions = self(continuous, categorical)
        loss = self.loss_fn(predictions, labels)

        opt_dense, opt_sparse = self.optimizers()
        opt_dense.zero_grad()
        self.manual_backward(loss)
        opt_dense.step()
        if opt_sparse:
            opt_sparse.zero_grad()
            opt_sparse.step()

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        continuous, categorical, labels = batch
        labels = labels.float().view(-1, 1)
        predictions = self(continuous, categorical)
        loss = self.loss_fn(predictions, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        dense_params, sparse_params = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                (sparse_params if "embedding" in name else dense_params).append(param)

        optimizers = []
        if dense_params:
            optimizers.append(torch.optim.Adam(dense_params, lr=self.lr))
        if sparse_params:
            optimizers.append(torch.optim.SparseAdam(sparse_params, lr=self.lr))
        return optimizers


def get_netflix_dataloader(csv_path="data/netflix_titles.csv", batch_size=32):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["release_year", "duration", "type", "rating"])

    def parse_duration(val):
        try:
            return int(val.split(" ")[0])
        except:
            return 0

    df["parsed_duration"] = df["duration"].apply(parse_duration)

    # Save artifacts for inference
    fit_and_save_scaler(df["release_year"].values, df["parsed_duration"].values)
    fit_and_save_encoders(df["type"], df["rating"])

    # Preprocess continuous features
    cont = StandardScaler().fit_transform(
        df[["release_year", "parsed_duration"]].values
    )
    continuous_tensor = torch.tensor(cont, dtype=torch.float32)

    # Preprocess categorical features
    type_idx = LabelEncoder().fit_transform(df["type"])
    rating_idx = LabelEncoder().fit_transform(df["rating"])
    categorical_tensor = torch.tensor(
        np.stack([type_idx, rating_idx], axis=1), dtype=torch.int64
    )

    # Dummy labels
    labels = torch.randint(0, 2, (len(df), 1), dtype=torch.float32)

    dataset = TensorDataset(continuous_tensor, categorical_tensor, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    checkpoint_callback = ModelCheckpoint(
        dirpath="lightning_logs/checkpoints",
        filename="dlrm-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )

    logger = TensorBoardLogger("lightning_logs", name="dlrm")

    dataloader = get_netflix_dataloader()

    model = DLRMTrainer(
        num_features=2,
        embedding_sizes=[2, 18],
        mlp_layers=[64, 32, 16],
    )

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="cpu",
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, dataloader)

    torch.save(model.model.state_dict(), "trained_model.pth")
    print("✅ Model trained and saved as trained_model.pth")
