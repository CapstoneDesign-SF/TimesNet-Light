import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time
from base.base_model import BaseDeepAD
from base.utility import get_sub_seqs
from .timesnetmodule import TimesBlock, DataEmbedding


class TimesNet(BaseDeepAD):
    def __init__(self, seq_len=100, stride=1, lr=0.0001, epochs=10, batch_size=32,
                 epoch_steps=20, prt_steps=1, device='cuda',
                 pred_len=0, e_layers=2, d_model=64, d_ff=64, dropout=0.1, top_k=5, num_kernels=6,
                 verbose=2, random_state=42, net=None):

        super(TimesNet, self).__init__(
            model_name='TimesNet', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )
        self.pred_len = pred_len
        self.e_layers = e_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.top_k = top_k
        self.num_kernels = num_kernels

        self.net = net

    def fit(self, X, y=None):
        self.n_features = X.shape[1]

        train_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)

        self.net = TimesNetModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=self.n_features,
            c_out=self.n_features,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            top_k=self.top_k,
            num_kernels=self.num_kernels
        ).to(self.device)

        dataloader = DataLoader(train_seqs, batch_size=self.batch_size,
                                shuffle=True, pin_memory=True)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.net.train()
        for e in range(self.epochs):
            t1 = time.time()
            loss = self.training(dataloader)

            if self.verbose >= 1 and (e == 0 or (e + 1) % self.prt_steps == 0):
                print(f'epoch{e + 1:3d}, '
                      f'training loss: {loss:.6f}, '
                      f'time: {time.time() - t1:.1f}s')
        self.decision_scores_ = self.decision_function(X)
        self.labels_ = self._process_decision_scores()  # in base model
        return

    def decision_function(self, X, return_rep=False):
        seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=1)
        dataloader = DataLoader(seqs, batch_size=self.batch_size,
                                shuffle=False, drop_last=False)

        self.net.eval()
        loss, _ = self.inference(dataloader)  # (n,d)
        loss_final = np.mean(loss, axis=1)  # (n,)

        padding_list = np.zeros([X.shape[0] - loss.shape[0], loss.shape[1]])
        loss_pad = np.concatenate([padding_list, loss], axis=0)
        loss_final_pad = np.hstack([0 * np.ones(X.shape[0] - loss_final.shape[0]), loss_final])

        return loss_final_pad

    def training(self, dataloader):
        criterion = nn.MSELoss()
        train_loss = []

        for ii, batch_x in enumerate(dataloader):
            self.optimizer.zero_grad()

            batch_x = batch_x.float().to(self.device)  # (bs, seq_len, dim)
            outputs = self.net(batch_x)  # (bs, seq_len, dim)
            loss = criterion(outputs[:, -1:, :], batch_x[:, -1:, :])
            train_loss.append(loss.item())

            loss.backward()
            self.optimizer.step()

            if self.epoch_steps != -1:
                if ii > self.epoch_steps:
                    break
        self.scheduler.step()

        return np.average(train_loss)

    def inference(self, dataloader):
        criterion = nn.MSELoss(reduction='none')
        attens_energy = []
        preds = []

        # with torch.no_gard():
        for batch_x in dataloader:  # test_set
            batch_x = batch_x.float().to(self.device)

            outputs = self.net(batch_x)
            # criterion
            score = criterion(batch_x[:, -1:, :], outputs[:, -1:, :]).squeeze(1)  # (bs, dim)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0)  # anomaly scores
        test_energy = np.array(attens_energy)  # anomaly scores

        return test_energy, preds

    def training_forward(self, batch_x, net, criterion):
        """define forward step in training"""
        return

    def inference_forward(self, batch_x, net, criterion):
        """define forward step in inference"""
        return

    def training_prepare(self, X, y):
        """define train_loader, net, and criterion"""
        return

    def inference_prepare(self, X):
        """define test_loader"""
        return


# proposed model
class TimesNetModel(nn.Module):

    def __init__(self, seq_len, pred_len, enc_in, c_out,
                 e_layers, d_model, d_ff, dropout, top_k, num_kernels):

        super(TimesNetModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = nn.ModuleList([TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
                                    for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc):
        dec_out = self.anomaly_detection(x_enc)
        return dec_out  # [B, L, D]