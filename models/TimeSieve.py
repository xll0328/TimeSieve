import torch
import torch.nn as nn
import torch.nn.functional as F

import pywt
import numpy as np

import torch.fft as fft

class WaveletDecomposition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, wavelet='db1', mode='symmetric'):
        cA_list = []
        cD_list = []

        for feature_idx in range(data.shape[2]):
            signal = data[:, :, feature_idx].detach().cpu().numpy()
            cA, cD = pywt.dwt(signal, wavelet, mode=mode)
            cA_list.append(cA)
            cD_list.append(cD)

        cA_tensor = torch.tensor(np.stack(cA_list, axis=-1), dtype=data.dtype, device=data.device)
        cD_tensor = torch.tensor(np.stack(cD_list, axis=-1), dtype=data.dtype, device=data.device)

        ctx.save_for_backward(data)
        ctx.wavelet = wavelet
        ctx.mode = mode

        return cA_tensor, cD_tensor

    @staticmethod
    def backward(ctx, grad_cA, grad_cD):
        data, = ctx.saved_tensors
        wavelet = ctx.wavelet
        mode = ctx.mode

        grad_data_list = []

        for feature_idx in range(data.shape[2]):
            cA_grad = grad_cA[:, :, feature_idx].detach().cpu().numpy()
            cD_grad = grad_cD[:, :, feature_idx].detach().cpu().numpy()
            reconstructed_signal = pywt.idwt(cA_grad, cD_grad, wavelet, mode=mode)
            grad_data_list.append(reconstructed_signal)

        grad_data = torch.tensor(np.stack(grad_data_list, axis=-1), dtype=data.dtype, device=data.device)

        return grad_data, None, None

def wavelet_decomposition(data, wavelet='db1', mode='symmetric'):
    return WaveletDecomposition.apply(data, wavelet, mode)


class InverseWaveletTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cA_tensor, cD_tensor, wavelet='db1', mode='symmetric'):
        cA_np = cA_tensor.detach().cpu().numpy()
        cD_np = cD_tensor.detach().cpu().numpy()

        reconstructed_list = []

        for feature_idx in range(cA_tensor.shape[2]):
            cA_trend = cA_np[:, :, feature_idx]
            cD_seasonal = cD_np[:, :, feature_idx]
            reconstructed_signal = pywt.idwt(cA_trend, cD_seasonal, wavelet, mode=mode)
            reconstructed_list.append(reconstructed_signal)

        reconstructed_data = torch.tensor(np.stack(reconstructed_list, axis=-1), dtype=cA_tensor.dtype, device=cA_tensor.device)

        ctx.save_for_backward(cA_tensor, cD_tensor)
        ctx.wavelet = wavelet
        ctx.mode = mode

        return reconstructed_data

    @staticmethod
    def backward(ctx, grad_output):
        cA_tensor, cD_tensor = ctx.saved_tensors
        wavelet = ctx.wavelet
        mode = ctx.mode

        grad_output_np = grad_output.detach().cpu().numpy()

        grad_cA_list = []
        grad_cD_list = []

        for feature_idx in range(cA_tensor.shape[2]):
            reconstructed_signal = grad_output_np[:, :, feature_idx]
            cA_grad, cD_grad = pywt.dwt(reconstructed_signal, wavelet, mode=mode)
            grad_cA_list.append(cA_grad)
            grad_cD_list.append(cD_grad)

        grad_cA_tensor = torch.tensor(np.stack(grad_cA_list, axis=-1), dtype=cA_tensor.dtype, device=cA_tensor.device)
        grad_cD_tensor = torch.tensor(np.stack(grad_cD_list, axis=-1), dtype=cA_tensor.dtype, device=cA_tensor.device)

        return grad_cA_tensor, grad_cD_tensor, None, None

def inverse_wavelet_transform(cA_tensor, cD_tensor, wavelet='db1', mode='symmetric'):
    return InverseWaveletTransform.apply(cA_tensor, cD_tensor, wavelet, mode)
class InformationBottleneck_trend(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InformationBottleneck_trend, self).__init__()

        self.in_d = input_dim
        self.out_d = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),#eLU(inplace=True),
            nn.Linear(output_dim, output_dim),
            #nn.Linear(output_dim*4, output_dim),

        )

        self.fc_mu = nn.Linear(output_dim, output_dim)
        self.fc_std = nn.Linear(output_dim, output_dim)

        self.decoder = nn.Linear(output_dim, input_dim)

    def encode(self, x, beta=1e-3):

        x_ = x
        x = self.encoder(x) # + x_
        return self.fc_mu(x), F.softplus(self.fc_std(x) - 5)

    def decode(self, z):

        return self.decoder(z)

    def reparameterize(self, mu, std):

        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, beta=1e-3):

        mu, std = self.encode(x, beta)
        logvar = torch.log(std.pow(2))
        z = self.reparameterize(mu, std)
        output = self.decode(z) + x
        return output, self.loss_function(output, x, mu, logvar, beta)*0.0001

    def loss_function(self, recon_x, x, mu, logvar, beta=1e-3):

        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss
        return loss


class InformationBottleneck(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InformationBottleneck, self).__init__()

        self.in_d = input_dim
        self.out_d = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),#eLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

        self.fc_mu = nn.Linear(output_dim, output_dim)
        self.fc_std = nn.Linear(output_dim, output_dim)

        self.decoder = nn.Linear(output_dim, input_dim)

    def encode(self, x, beta=1e-3):
        x_ = x
        x = self.encoder(x) # + x_
        return self.fc_mu(x), F.softplus(self.fc_std(x) - 5)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, beta=1e-3):
        mu, std = self.encode(x, beta)
        logvar = torch.log(std.pow(2))
        z = self.reparameterize(mu, std)
        output = self.decode(z) + x
        return output, self.loss_function(output, x, mu, logvar, beta)*0.0001

    def loss_function(self, recon_x, x, mu, logvar, beta=1e-3):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss
        return loss

class Model(nn.Module):
    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.dec_in = configs.dec_in
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.decompsition = wavelet_decomposition
        self.IDWT = inverse_wavelet_transform
        self.individual = individual
        self.channels = configs.enc_in
        self.IB = InformationBottleneck(input_dim=self.pred_len, output_dim=self.pred_len)
        self.IB_trend = InformationBottleneck_trend(input_dim=self.pred_len, output_dim=self.pred_len)
        self.pre_1 = nn.Linear(self.seq_len, self.pred_len*5)
        self.pre_2 = nn.Linear(self.pred_len*5, self.pred_len*10)
        self.pre_3 = nn.Linear(self.pred_len*10, self.pred_len)
        self.relu = nn.ReLU()
        # self.mlp = MLP(input_dim=48, hidden_dim=96, output_dim=48)

        # MLP 配置
        self.mlp_input_dim = int(self.seq_len/2)
        self.mlp_hidden_dim = self.seq_len
        self.mlp_output_dim = int(self.seq_len/2)

        self.imlp_input_dim = self.dec_in
        self.imlp_hidden_dim = self.dec_in * 2
        self.imlp_output_dim = self.dec_in

        self.out_mlp_input_dim = self.seq_len
        self.out_mlp_hidden_dim = self.seq_len * 5
        self.out_mlp_output_dim = self.pred_len


        self.out_mlp = nn.Sequential(
            nn.Linear(self.out_mlp_input_dim, self.out_mlp_hidden_dim * 2),

            nn.Linear(self.out_mlp_hidden_dim * 2, self.out_mlp_hidden_dim * 4),
            nn.Linear(self.out_mlp_hidden_dim * 4, self.out_mlp_hidden_dim * 8),

            nn.Linear(self.out_mlp_hidden_dim * 8, self.out_mlp_output_dim)
        )




        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        trend_init, seasonal_init = self.decompsition(data=x)
        trend_out, seasonal_out = trend_init.permute(0, 2, 1), seasonal_init.permute(0,2, 1)
        seasonal_output, loss_sea = self.IB(seasonal_out)
        trend_output, loss_trend = self.IB_trend(trend_out)
        trend_output, seasonal_output = trend_output.permute(0, 2, 1), seasonal_output.permute(0, 2, 1)

        final_output_ = self.IDWT(trend_init + trend_output, seasonal_init+seasonal_output)
        final_output =final_output_.permute(0, 2, 1)
        final_output = self.out_mlp(final_output)
        return final_output.permute(0, 2, 1), loss_trend+loss_sea, final_output_

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, loss, final_output = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :], loss, final_output  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
