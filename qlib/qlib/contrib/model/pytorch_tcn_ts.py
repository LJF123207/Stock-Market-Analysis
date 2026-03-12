# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from .tcn import TemporalConvNet
from .CrossAttention import CrossAttentionTSNews
from .Decoder import CrossAttentionDecoder


class TCN(Model):
    """TCN Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        n_chans=128,
        kernel_size=5,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("TCN")
        self.logger.info("TCN pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.n_chans = n_chans
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed
        self.usenews = bool(kwargs.pop("usenews", False))
        self.fusion_type = str(kwargs.pop("fusion_type", "add"))
        self.layer_num = int(kwargs.pop("layer_num", 2))
        self.news_dim = int(kwargs.pop("d_news", 0)) if self.usenews else 0

        self.logger.info(
            "TCN parameters setting:"
            "\nd_feat : {}"
            "\nn_chans : {}"
            "\nkernel_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nseed : {}"
            "\nusenews : {}"
            "\nfusion_type : {}"
            "\nlayer_num : {}"
            "\nd_news : {}".format(
                d_feat,
                n_chans,
                kernel_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                n_jobs,
                self.use_gpu,
                seed,
                self.usenews,
                self.fusion_type,
                self.layer_num,
                self.news_dim,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.TCN_model = TCNModel(
            num_input=self.d_feat,
            output_size=1,
            num_channels=[self.n_chans] * self.num_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            d_news=self.news_dim,
            use_news=self.usenews,
            fusion_type=self.fusion_type,
            layer_num=self.layer_num,
        )
        self.logger.info("model:\n{:}".format(self.TCN_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.TCN_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.TCN_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.TCN_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.TCN_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):
        self.TCN_model.train()

        for data in data_loader:
            feature, label = self._prepare_input(data)
            pred = self.TCN_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.TCN_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.TCN_model.eval()

        scores = []
        losses = []

        for data in data_loader:
            feature, label = self._prepare_input(data)
            with torch.no_grad():
                pred = self.TCN_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
    ):
        train_cols = ["feature", "label"]
        if self.usenews:
            train_cols = ["feature", "news", "label"]
        dl_train = dataset.prepare("train", col_set=train_cols, data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=train_cols, data_key=DataHandlerLP.DK_L)

        # process nan brought by dataloader
        dl_train.config(fillna_type="ffill+bfill")
        # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")

        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs, drop_last=True
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.TCN_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.TCN_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        test_cols = ["feature", "label"]
        if self.usenews:
            test_cols = ["feature", "news", "label"]
        dl_test = dataset.prepare("test", col_set=test_cols, data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.TCN_model.eval()
        preds = []

        for data in test_loader:
            feature, _ = self._prepare_input(data)
            with torch.no_grad():
                pred = self.TCN_model(feature).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())

    def _prepare_input(self, data):
        raw = data.to(self.device)
        feat = raw[:, :, : self.d_feat]
        if self.usenews and self.news_dim > 0:
            if self.fusion_type == "concat":
                news = raw[:, :, self.d_feat : self.d_feat + self.news_dim]
                feat = torch.cat([feat, news.float()], dim=-1)
                feat = self.TCN_model.fusion_proj(feat.float())
            elif self.fusion_type == "add":
                news = raw[:, :, self.d_feat : self.d_feat + self.news_dim]
                feat = feat + self.TCN_model.news_proj(news.float())
            elif self.fusion_type == "crossattn":
                news = raw[:, :, self.d_feat : self.d_feat + self.news_dim]
                for layer in self.TCN_model. cross_attn_layers:
                    feat, _ = layer(feat.float(), news.float())
            elif self.fusion_type == "decoder":
                news = raw[:, :, self.d_feat : self.d_feat + self.news_dim]
                feat = self.TCN_model.decoder_layers(feat.float(), self.TCN_model.news_proj(news.float()))
        feature = torch.transpose(feat, 1, 2).float()
        label = raw[:, -1, -1]
        return feature, label


class TCNModel(nn.Module):
    def __init__(self, num_input, output_size, num_channels, kernel_size, dropout, d_news=0, use_news=False, fusion_type="add", layer_num=2):
        super().__init__()
        self.num_input = num_input
        self.use_news = use_news and d_news > 0
        self.fusion_type = fusion_type
        self.layer_num = layer_num
        self.news_proj = nn.Linear(d_news, num_input) if self.use_news  else None
        self.fusion_proj = nn.Linear(num_input + d_news, num_input) if self.use_news and self.fusion_type == "concat" else None
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        if self.use_news and self.fusion_type == "crossattn":
            self.cross_attn_layers = nn.ModuleList(
                [
                    CrossAttentionTSNews(
                        d_model=self.num_input,
                        d_news=d_news,
                        n_heads=4,
                    )
                    for _ in range(self.layer_num)
                ]
            )
        if self.use_news and self.fusion_type == "decoder":
            self.decoder_layers = CrossAttentionDecoder(
                d_model=self.num_input,
                nhead=8,
                num_layers=self.layer_num,
                dim_feedforward=4 * self.num_input,
                dropout=0.1,
            )
        
    def forward(self, x):
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        return output.squeeze()
