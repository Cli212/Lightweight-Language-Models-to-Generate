import torch

from typing import Any, Dict, List

import math
import warnings
from functools import partial

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import numpy as np
    import pytorch_lightning as pl
    import torch
    import torch.nn as nn
    # from pytorch_lightning.metrics.functional import accuracy, auroc
    # from pytorch_lightning.metrics.functional import f1 as f1_score
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    try:
        from torchqrnn import QRNN
    except ImportError:
        print("Import QRNN from torchqrnn fail")
        from torch.nn import LSTM as QRNN


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PQRNN(pl.LightningModule):
    def __init__(
        self,
        b: int = 512,
        d: int = 64,
        num_layers: int = 4,
        fc_sizes: List[int] = None,
        output_size: int = 2,
        lr: float = 0.025,
        dropout: float = 0.5,
        rnn_type: str = "LSTM",
        multilabel: bool = False,
        nhead: int = 8,
    ):
        super().__init__()
        if fc_sizes is None:
            fc_sizes = [128, 64]

        self.hyparams: Dict[str, Any] = {
            "b": b,
            "d": d,
            "fc_size": fc_sizes,
            "lr": lr,
            "output_size": output_size,
            "dropout": dropout,
            "rnn_type": rnn_type.upper(),
            "multilabel": multilabel,
            "nhead": nhead,
            "n_layers": num_layers
        }

        layers: List[nn.Module] = []
        for x, y in zip([d] + fc_sizes, fc_sizes + [output_size]):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(x, y))

        self.tanh = nn.Hardtanh()
        if self.hyparams["rnn_type"] in {"LSTM", "GRU", "QRNN"}:
            self.hidden = {
                "LSTM": partial(nn.LSTM, bidirectional=True, batch_first=True, num_layers=self.hyparams['n_layers']),
                "GRU": partial(nn.GRU, bidirectional=True, batch_first=True, num_layers=self.hyparams['n_layers']),
                "QRNN": QRNN,
            }[self.hyparams["rnn_type"]](
                b*2, d, num_layers=num_layers, dropout=dropout
            )
        else:
            self.pos_encoder = PositionalEncoding(d_model=b, dropout=dropout)
            encoder_layers = TransformerEncoderLayer(
                d_model=b, nhead=nhead, dropout=dropout
            )
            self.hidden = TransformerEncoder(
                encoder_layers, num_layers=num_layers
            )
            self.linear = nn.Linear(b, d)

        self.output = nn.ModuleList(layers)
        self.loss = (
            nn.CrossEntropyLoss()
            if not self.hyparams["multilabel"]
            else nn.BCEWithLogitsLoss()
        )

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, projection, hidden=None):
        features = self.tanh(projection)
        # features = features.transpose(0, 1)
        if self.hyparams["rnn_type"] in {"LSTM", "GRU", "QRNN"}:
            output, hidden = self.hidden(features, hidden)
            # if self.hyparams["rnn_type"] != "QRNN":
            #     output = (
            #         output.T[..., : output.shape[-1] // 2]
            #         + output[..., output.shape[-1] // 2 :]
            #     )
        else:
            features = features * math.sqrt(self.hyparams["b"])
            features = self.pos_encoder(features)
            output = self.hidden(
                features,
                self.generate_square_subsequent_mask(features.size(0)).to(
                    features.device
                ),
            )
            output = self.linear(output)

        # Sum bidirectional GRU outputs
        output = output[:, :, :self.hyparams['d']] + output[:, :, self.hyparams['d']:]
        # Return output and final hidden state
        return output, hidden

    def training_step(self, batch, batch_idx):
        projection, _, labels = batch
        logits = self.forward(projection)
        self.log(
            "loss",
            self.loss(
                logits,
                labels.type(
                    logits.dtype if self.hyparams["multilabel"] else labels.dtype
                ),
            )
            .detach()
            .cpu()
            .item(),
        )
        return {
            "loss": self.loss(
                logits,
                labels.type(
                    logits.dtype if self.hyparams["multilabel"] else labels.dtype
                ),
            )
        }

    def validation_step(self, batch, batch_idx):
        projection, _, labels = batch
        logits = self.forward(projection)

        return {"logits": logits, "labels": labels}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyparams["lr"])
        scheduler = ReduceLROnPlateau(optimizer, "min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

