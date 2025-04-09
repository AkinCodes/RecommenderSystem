import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import TensorBoardLogger
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.dlrm import DLRMModel


class DLRMTrainer(pl.LightningModule):
    def __init__(self, num_features, embedding_sizes, mlp_layers, lr=1e-3):
        super(DLRMTrainer, self).__init__()

        self.model = DLRMModel(num_features, embedding_sizes, mlp_layers)
        self.loss_fn = nn.BCELoss()
        self.lr = lr

        self.automatic_optimization = False

    def forward(self, continuous_features, categorical_features):
        return self.model(continuous_features, categorical_features)

    def training_step(self, batch, batch_idx):
        continuous_features, categorical_features, labels = batch
        labels = labels.float().view(-1, 1)

        predictions = self(continuous_features, categorical_features)
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
        continuous_features, categorical_features, labels = batch
        labels = labels.float().view(-1, 1)
        predictions = self(continuous_features, categorical_features)
        loss = self.loss_fn(predictions, labels)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer_params = []
        sparse_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if "embedding" in name:
                    sparse_params.append(param)
                else:
                    optimizer_params.append(param)

        optimizers = []
        if optimizer_params:
            optimizers.append(optim.Adam(optimizer_params, lr=self.lr))
        if sparse_params:
            optimizers.append(optim.SparseAdam(sparse_params, lr=self.lr))

        return optimizers


def get_dataloader():
    dataset = TensorDataset(
        torch.randn(1000, 10),
        torch.randint(0, 5, (1000, 5)),
        torch.randint(0, 2, (1000, 1)),
    )
    return DataLoader(dataset, batch_size=32, shuffle=True)


if __name__ == "__main__":
    from pytorch_lightning.callbacks import ModelCheckpoint

    logging.basicConfig(level=logging.DEBUG)

    checkpoint_callback = ModelCheckpoint(
        dirpath="lightning_logs/checkpoints",
        filename="dlrm-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )

    tensorboard_logger = TensorBoardLogger("lightning_logs", name="dlrm")

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        enable_checkpointing=True,
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback],
    )

    model = DLRMTrainer(
        num_features=10, embedding_sizes=[10, 10, 10, 10, 10], mlp_layers=[64, 32, 16]
    )

    # Test Model Forward Pass BEFORE Training
    batch = next(iter(get_dataloader()))
    continuous_features, categorical_features, labels = batch
    output = model(continuous_features, categorical_features)

    # Train Model
    trainer.fit(model, get_dataloader())

    # Validation After Training
    print("Running Validation...")
    trainer.validate(model, get_dataloader())

    # Load Model from Checkpoint
    checkpoint_path = "lightning_logs/checkpoints/dlrm-epoch=04-train_loss=0.65.ckpt"
    model = DLRMTrainer.load_from_checkpoint(
        checkpoint_path,
        num_features=10,
        embedding_sizes=[10, 10, 10, 10, 10],
        mlp_layers=[64, 32, 16],
    )

    # Run Inference with Loaded Model
    dataloader = get_dataloader()
    batch = next(iter(dataloader))

    continuous_features, categorical_features, labels = batch
    output = model(continuous_features, categorical_features)
    print(f"Loaded Model Output Shape: {output.shape}")
