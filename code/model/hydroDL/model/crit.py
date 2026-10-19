import torch
import numpy as np
import math


class SigmaLoss(torch.nn.Module):
    def __init__(self, prior='gauss'):
        super(SigmaLoss, self).__init__()
        self.reduction = 'elementwise_mean'
        if prior == '':
            self.prior = None
        else:
            self.prior = prior.split('+')

    def forward(self, output, target):
        ny = target.shape[-1]
        lossMean = 0
        for k in range(ny):
            p0 = output[:, :, k * 2]
            s0 = output[:, :, k * 2 + 1]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            s = s0[mask]
            t = t0[mask]
            if self.prior[0] == 'gauss':
                loss = torch.exp(-s).mul((p - t)**2) / 2 + s / 2
            elif self.prior[0] == 'invGamma':
                c1 = float(self.prior[1])
                c2 = float(self.prior[2])
                nt = p.shape[0]
                loss = torch.exp(-s).mul(
                    (p - t)**2 + c2 / nt) / 2 + (1 / 2 + c1 / nt) * s
            lossMean = lossMean + torch.mean(loss)
        return lossMean


class RmseLoss(torch.nn.Module):
    def __init__(self):
        super(RmseLoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp
        return loss

class RmseLossRunoff(torch.nn.Module):
    def __init__(self):
        super(RmseLossRunoff, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = torch.log10(torch.sqrt(output[:, :, k]) + 0.1)
            t0 = torch.log10(torch.sqrt(target[:, :, k]) + 0.1)
            # p0 = torch.sqrt(output[:, :, k])
            # t0 = torch.sqrt(target[:, :, k])
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp
        return loss

class RmseLossComb(torch.nn.Module):
    def __init__(self, alpha, beta=1e-6):
        super(RmseLossComb, self).__init__()
        self.alpha = alpha # weights of log-sqrt RMSE
        self.beta = beta

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(output[:, :, k]+self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k]+self.beta) + 0.1)
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss1 = torch.sqrt(((p - t)**2).mean()) # RMSE item
            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            # pa = torch.log10(torch.sqrt(p) + 0.1)
            # ta = torch.log10(torch.sqrt(t) + 0.1)
            loss2 = torch.sqrt(((pa - ta)**2).mean()) #Log-Sqrt RMSE item
            temp = (1.0-self.alpha)*loss1 + self.alpha*loss2
            loss = loss + temp
        return loss

class RmseLossComb_penalty(torch.nn.Module):
    def __init__(self, alpha, beta=1e-6):
        super(RmseLossComb_penalty, self).__init__()
        self.alpha = alpha # weights of log-sqrt RMSE
        self.beta = beta

    def forward(self, output, target, delta_para):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            p1 = torch.log10(torch.sqrt(output[:, :, k]+self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(target[:, :, k]+self.beta) + 0.1)
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            loss1 = torch.sqrt(((p - t)**2).mean()) # RMSE item
            mask1 = t1 == t1
            pa = p1[mask1]
            ta = t1[mask1]
            # pa = torch.log10(torch.sqrt(p) + 0.1)
            # ta = torch.log10(torch.sqrt(t) + 0.1)
            loss2 = torch.sqrt(((pa - ta)**2).mean()) #Log-Sqrt RMSE item
            temp = (1.0-self.alpha)*loss1 + self.alpha*loss2
            loss = loss + temp + delta_para*10
            print(delta_para)
            print(loss)
        return loss


class RmseLossCNN(torch.nn.Module):
    def __init__(self):
        super(RmseLossCNN, self).__init__()

    def forward(self, output, target):
        # output = ngrid * nvar * ntime
        ny = target.shape[1]
        loss = 0
        for k in range(ny):
            p0 = output[:, k, :]
            t0 = target[:, k, :]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp
        return loss

class RmseLossANN(torch.nn.Module):
    def __init__(self, get_length=False):
        super(RmseLossANN, self).__init__()
        self.ind = get_length

    def forward(self, output, target):
        if len(output.shape) == 2:
            p0 = output[:, 0]
            t0 = target[:, 0]
        else:
            p0 = output[:, :, 0]
            t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        loss = torch.sqrt(((p - t)**2).mean())
        if self.ind is False:
            return loss
        else:
            Nday = p.shape[0]
            return loss, Nday

class ubRmseLoss(torch.nn.Module):
    def __init__(self):
        super(ubRmseLoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            pmean = p.mean()
            tmean = t.mean()
            p_ub = p-pmean
            t_ub = t-tmean
            temp = torch.sqrt(((p_ub - t_ub)**2).mean())
            loss = loss + temp
        return loss

class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = ((p - t)**2).mean()
            loss = loss + temp
        return loss

class NSELoss(torch.nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii, 0]
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask==True])>0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                if SST != 0:
                    SSRes = torch.sum((t - p) ** 2)
                    temp = 1 - SSRes / SST
                    losssum = losssum + temp
                    nsample = nsample +1
        # minimize the opposite average NSE
        loss = -(losssum/nsample)
        return loss

class NSELosstest(torch.nn.Module):
    # Same as Fredrick 2019
    def __init__(self):
        super(NSELosstest, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii, 0]
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask==True])>0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                SSRes = torch.sum((t - p) ** 2)
                # temp = SSRes / ((torch.sqrt(SST)+0.1)**2)
                temp = SSRes / (SST+0.1)
                losssum = losssum + temp
                nsample = nsample +1
        loss = losssum/nsample
        return loss

class NSELossBatch(torch.nn.Module):
    # Same as Fredrick 2019, batch NSE loss
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray, eps=0.1):
        super(NSELossBatch, self).__init__()
        self.std = stdarray
        self.eps = eps

    def forward(self, output, target, igrid):
        nt = target.shape[0]
        stdse = np.tile(self.std[igrid].T, (nt, 1))
        stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()
        p0 = output[:, :, 0]   # dim: Time*Gage
        t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        stdw = stdbatch[mask]
        sqRes = (p - t)**2
        normRes = sqRes / (stdw + self.eps)**2
        loss = torch.mean(normRes)

        # sqRes = (t0 - p0)**2 # squared error
        # normRes = sqRes / (stdbatch + self.eps)**2
        # mask = t0 == t0
        # loss = torch.mean(normRes[mask])
        return loss

class NSELossBatch_song(torch.nn.Module):
    # Same as Fredrick 2019, batch NSE loss
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray, eps=0.1):
        super(NSELossBatch_song, self).__init__()
        self.std = stdarray
        self.eps = eps

    def forward(self, output, target, igrid):
        nt = target.shape[0]
        stdse = np.tile(self.std[igrid].T, (nt, 1))
        stdbatch = torch.tensor(stdse, requires_grad=False).float().to(output)
        p0 = output[:, :, 0]   # dim: Time*Gage
        t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        stdw = stdbatch[mask]
        sqRes = (p - t)**2
        normRes = sqRes / (stdw + self.eps)**2
        loss = torch.mean(normRes)

        # sqRes = (t0 - p0)**2 # squared error
        # normRes = sqRes / (stdbatch + self.eps)**2
        # mask = t0 == t0
        # loss = torch.mean(normRes[mask])
        return loss

class NSELossBatch_li(torch.nn.Module):
    # Same as Fredrick 2019, batch NSE loss
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray, eps=0.1):
        super(NSELossBatch_li, self).__init__()
        self.std = stdarray
        self.eps = eps

    def forward(self, output, target, igrid):
        if len(np.shape(output))>2:
            nt = target.shape[0]
            stdse = np.tile(self.std[igrid][None,:], (nt, 1,1))
            stdbatch = torch.tensor(stdse, requires_grad=False).float().to(output)
        else:
            stdse=np.tile(self.std[igrid][None,:],(nt,1,1))
            stdbatch=torch.tensor(stdse,requires_grad=False).float().to(output)
        if len(np.shape(output))<3:
            p0=output[:,:target.shape[1]]  # dim: Time*Gage
            t0=target[:,:]
        else:
            p0 = output[:, :, :target.shape[2]]   # dim: Time*Gage
            t0 = target[:, :, :]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        stdw = stdbatch[mask]
        sqRes = (p - t)**2
        normRes = sqRes / (stdw + self.eps)**2
        loss = torch.mean(normRes)

        # sqRes = (t0 - p0)**2 # squared error
        # normRes = sqRes / (stdbatch + self.eps)**2
        # mask = t0 == t0
        # loss = torch.mean(normRes[mask])
        return loss


def corr_loss(p2,t02,nvv):
    # p2, t02: [365, 30]
    corr_list=[]
    for i in range(p2.shape[1]):
        x=p2[:,i]
        y=t02[:,i]

        # Find positions that are not NaN
        mask=~torch.isnan(y)
        if mask.sum()<nvv:
            continue  # Cannot compute the correlation coefficient; skip

        x_valid=x[mask]
        y_valid=y[mask]

        # Normalize
        x_mean=x_valid.mean()
        y_mean=y_valid.mean()
        x_std=x_valid.std()
        y_std=y_valid.std()

        if x_std==0 or y_std==0:
            continue  # Avoid division by zero or an invalid standard deviation

        corr=((x_valid-x_mean)*(y_valid-y_mean)).mean()/(x_std*y_std)
        corr_list.append(corr)

    if len(corr_list)==0:
        return torch.tensor(float('nan'))  # No columns can be computed
        # return torch.tensor(0.0)  # No columns can be computed


    corr_tensor=torch.stack(corr_list)
    loss=1-corr_tensor.mean()  # Use the negative correlation coefficient as the loss
    return loss




def corr_loss_trim(p2,t02,nvv,trim):
    # p2, t02: [365, 30]
    corr_list=[]
    for i in range(p2.shape[1]):
        x=p2[:,i]
        y=t02[:,i]

        # Find positions that are not NaN
        mask=~torch.isnan(y)
        if mask.sum()<nvv:
            continue  # Cannot compute the correlation coefficient; skip

        x_valid=x[mask]
        y_valid=y[mask]

        # Normalize
        x_mean=x_valid.mean()
        y_mean=y_valid.mean()
        x_std=x_valid.std()
        y_std=y_valid.std()

        if x_std==0 or y_std==0:
            continue  # Avoid division by zero or an invalid standard deviation

        corr=((x_valid-x_mean)*(y_valid-y_mean)).mean()/(x_std*y_std)
        corr_list.append(corr)

    if len(corr_list)==0:
        return torch.tensor(float('nan'))  # No columns can be computed
        # return torch.tensor(0.0)  # No columns can be computed


    corr_tensor=torch.stack(corr_list)

    # # Set a lower-tail percentile threshold, e.g., remove the worst 10%
    # low_percentile=0.1
    # low_threshold=torch.quantile(corr_tensor,low_percentile)

    # Set the trimming proportion, e.g., remove the lowest 10%
    p=trim
    sorted_corrs,_=torch.sort(corr_tensor)  # ascending order
    k=int(len(sorted_corrs)*p)  # Corresponding index
    low_threshold=sorted_corrs[k]  # Lower-tail threshold

    # Keep values greater than the threshold
    corrs_filtered=corr_tensor[corr_tensor>low_threshold]

    # Compute the mean
    mean_corr_filtered=torch.mean(corrs_filtered)

    loss=1-mean_corr_filtered # Use the negative correlation coefficient as the loss
    return loss


def corr_loss_trim_fast(p2,t02,nvv,trim):
    device=p2.device

    # 1. Build a strict mask
    # Count a value only when both the prediction and target are not NaN
    mask=(~torch.isnan(t02)&~torch.isnan(p2))

    # Fill invalid positions with 0 to avoid interfering with the sum
    # Note: do not use clone().masked_fill_; where is faster and more compatible
    x_fixed=torch.where(mask,p2,torch.zeros_like(p2))
    y_fixed=torch.where(mask,t02,torch.zeros_like(t02))

    # Compute the number of valid samples in each column
    n=mask.float().sum(dim=0)

    # Select valid columns: sample count >= nvv and sample count > 1 (because std requires n-1 > 0)
    valid_cols=(n>=nvv)&(n>1)

    if not valid_cols.any():
        return torch.tensor(float('nan'),device=device)

    # Use only valid columns for vectorized computation
    x=x_fixed[:,valid_cols]
    y=y_fixed[:,valid_cols]
    m=mask[:,valid_cols].float()
    nn=n[valid_cols]

    # 2. Compute the mean (using only entries where mask=1)
    x_mean=x.sum(dim=0)/nn
    y_mean=y.sum(dim=0)/nn

    # 3. Compute deviations (important: apply the mask again after subtracting the mean)
    x_diff=(x-x_mean)*m
    y_diff=(y-y_mean)*m

    # 4. Compute covariance and variance (strictly matching the original .std() denominator n-1)
    # Covariance formula: sum((x-mx)*(y-my)) / (n-1)
    cov=(x_diff*y_diff).sum(dim=0)/(nn-1)

    # Standard deviation formula: sqrt( sum((x-mx)^2) / (n-1) )
    x_var=(x_diff**2).sum(dim=0)/(nn-1)
    y_var=(y_diff**2).sum(dim=0)/(nn-1)

    x_std=torch.sqrt(x_var)
    y_std=torch.sqrt(y_var)

    # 5. Compute correlations and exclude columns with std=0
    eps=1e-12
    corr_mask=(x_std>eps)&(y_std>eps)

    if not corr_mask.any():
        return torch.tensor(float('nan'),device=device)

    final_corrs=cov[corr_mask]/(x_std[corr_mask]*y_std[corr_mask])

    # 6. Trimming logic: must exactly match the original [corr > low_threshold]
    sorted_corrs,_=torch.sort(final_corrs)
    k=int(len(sorted_corrs)*trim)
    low_threshold=sorted_corrs[k]

    # Original: corrs_filtered = corr_tensor[corr_tensor > low_threshold]
    corrs_filtered=final_corrs[final_corrs>low_threshold]

    # If all values are equal, the original > produces an empty result; provide a fallback here
    if corrs_filtered.numel()==0:
        mean_corr=sorted_corrs[k:].mean()
    else:
        mean_corr=corrs_filtered.mean()

    return 1-mean_corr


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def neural_sort(scores, tau=1.0):
#     """
#     NeuralSort: Differentiable sorting operator
#     Input: scores (batch, n)
#     Output: soft permutation matrix (batch, n, n)
#     """
#     n = scores.size(-1)
#     one = torch.ones((n, 1), device=scores.device)
#     scores = scores.unsqueeze(-1)  # (B, n, 1)
#
#     # pairwise difference matrix
#     diff = torch.abs(scores - scores.transpose(1,2))  # (B,n,n)
#
#     # scaling matrix
#     K = n + 1 - 2 * (torch.arange(n, device=scores.device).float() + 1)
#     K = K.view(1, 1, -1)  # (1,1,n)
#
#     P_max = torch.matmul(scores, K) - diff.sum(dim=-1, keepdim=True)  # (B,n,1)
#     P_hat = F.softmax(P_max / tau, dim=-2)  # (B,n,n)
#
#     return P_hat

# def soft_spearman_loss_vec(y_pred, y_true, tau=0.1, trim=0.0, nvv=5):
#     """
#     y_pred: (T, n, 1)
#     y_true: (T, n, 1)
#     trim: [0,1), trimming proportion; e.g., 0.1 removes the lowest 10%
#     nvv: minimum valid length
#     """
#     B, n, temp = y_pred.shape
#     corr_list = []
#
#     for i in range(n):  # by column
#         x = y_pred[:, i]
#         y = y_true[:, i]
#
#         # valid mask
#         mask = ~torch.isnan(y)
#         if mask.sum() < nvv:
#             continue
#
#         x_valid = x[mask].unsqueeze(0)  # (1, L)
#         y_valid = y[mask].unsqueeze(0)
#
#         # true rank
#         y_true_rank = torch.argsort(torch.argsort(y_valid, dim=1), dim=1).float()
#
#         # soft permutation matrix
#         P_hat = neural_sort(x_valid, tau=tau)  # (1, L, L)
#
#         # soft ranks
#         ranks = torch.arange(1, y_valid.shape[1] + 1, device=y_pred.device).float()
#         soft_rank = torch.matmul(P_hat.transpose(1, 2), ranks)
#
#         # Compute the Spearman surrogate
#         vx = soft_rank - soft_rank.mean(dim=1, keepdim=True)
#         vy = y_true_rank - y_true_rank.mean(dim=1, keepdim=True)
#         corr = (vx * vy).sum(dim=1) / (torch.norm(vx, dim=1) * torch.norm(vy, dim=1) + 1e-8)
#
#         corr_list.append(corr.squeeze())
#
#     if len(corr_list) == 0:
#         return torch.tensor(float('nan'), device=y_pred.device)
#
#     corr_tensor = torch.stack(corr_list)  # (n_valid,)
#
#     # ====== Add the trimming step ======
#     if trim > 0:
#         sorted_corrs, _ = torch.sort(corr_tensor)  # ascending
#         k = int(len(sorted_corrs) * trim)
#         if k > 0:
#             low_threshold = sorted_corrs[k]
#             corr_tensor = corr_tensor[corr_tensor > low_threshold]
#
#     mean_corr = corr_tensor.mean()
#     loss = 1 - mean_corr
#     return loss

import torch
import torch.nn.functional as F

def neural_sort(scores, tau=0.1):
    """
    NeuralSort: differentiable soft ranks
    scores: (L,) or (B,L)
    Returns: soft ranks, same shape
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)  # (1,L)
    B, L = scores.shape
    scores = scores.unsqueeze(-1)  # (B,L,1)

    # pairwise difference
    diff = torch.abs(scores - scores.transpose(1,2))  # (B,L,L)

    # scaling vector
    K = L + 1 - 2 * (torch.arange(L, device=scores.device).float() + 1)
    K = K.view(1,1,-1)

    P_max = torch.matmul(scores, K) - diff.sum(dim=-1, keepdim=True)
    P_hat = F.softmax(P_max / tau, dim=1)

    ranks = torch.arange(1, L+1, device=scores.device).float()
    soft_rank = torch.matmul(P_hat.transpose(1,2), ranks)  # (B,L)
    return soft_rank.squeeze(0) if soft_rank.shape[0]==1 else soft_rank


def soft_spearman_loss_parallel_nopad(y_pred, y_true, tau=0.1, trim=0.0, nvv=5, eps=1e-4):
    """
    GPU-parallel Soft Spearman loss, without padding, batched by valid length
    y_pred, y_true: (T,N)
    tau: soft rank temperature
    trim: Remove the lowest trim proportion
    nvv: minimum valid length
    eps: avoid division by zero
    """
    # T, N = y_pred.shape
    mask = ~torch.isnan(y_true)
    valid_counts = mask.sum(dim=0)
    valid_cols = torch.nonzero(valid_counts >= nvv).squeeze()

    if valid_cols.numel() == 0:
        return torch.tensor(float('nan'), device=y_pred.device)

    if valid_cols.dim() == 0:
        valid_cols = valid_cols.unsqueeze(0)
    valid_cols = valid_cols.tolist()

    corr_list = []

    for i in valid_cols:
        m = mask[:, i]
        x = y_pred[:, i][m]
        y = y_true[:, i][m]

        # Skip columns that are too short
        if x.numel() < 2:
            continue

        # Normalize inputs to prevent softmax numerical instability
        x = (x - x.mean()) / (x.std() + eps)

        # soft rank
        sr = neural_sort(x, tau=tau)
        yrank = torch.argsort(torch.argsort(y, dim=0), dim=0).float()

        # Compute the Pearson correlation
        vx = sr - sr.mean()
        vy = yrank - yrank.mean()
        corr = (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + eps)
        corr_list.append(corr)

    if len(corr_list) == 0:
        return torch.tensor(float('nan'), device=y_pred.device)

    corr_tensor = torch.stack(corr_list)

    # Trim the lower portion
    if trim > 0:
        sorted_corrs, _ = torch.sort(corr_tensor)
        k = int(len(sorted_corrs) * trim)
        if k > 0:
            low_thresh = sorted_corrs[k]
            corr_tensor = corr_tensor[corr_tensor > low_thresh]

    loss = 1 - corr_tensor.mean()
    return loss

def trimmed_mean(x, trim=0.3):
    """
    x: 1D tensor, may contain NaN
    trim: remove the lowest trim proportion
    Return the trimmed mean
    """
    # Remove NaNs first
    x_valid = x[~torch.isnan(x)]
    if x_valid.numel() == 0:
        return torch.tensor(float('nan'), device=x.device)

    # Sort
    x_sorted, _ = torch.sort(x_valid)

    # Compute the element index after removing the first trim proportion
    n = x_sorted.numel()
    k = int(n * trim)  # Remove the first k smallest values
    if k >= n:  # Avoid removing all values
        k = n - 1
    trimmed = x_sorted[k:]

    return trimmed.mean()

# def pairwise_monotone_loss_rowwise_sign(y_pred, y_true, trim=0.3, nvv=10, eps=1e-6):
#     """
#     Vectorized, safe version
#     Use the sign function instead of sigmoid:
#         diff>0 -> 1
#         diff<0 -> -1
#         diff=0 -> 0
# # sign outputs discrete values and is not continuously differentiable → gradients are nearly zero
# # Using it as a training loss prevents the model from learning through backpropagation
#     """
#     mask = (~torch.isnan(y_pred)) & (~torch.isnan(y_true))  # (N, T)
#
#     # Pairwise valid mask
#     pairwise_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (N, T, T)
#
#     # Keep only valid elements and fill the rest with 0
#     y_pred_safe = torch.where(mask, y_pred, torch.zeros_like(y_pred))
#     y_true_safe = torch.where(mask, y_true, torch.zeros_like(y_true))
#
#     diff_pred = y_pred_safe.unsqueeze(2) - y_pred_safe.unsqueeze(1)
#     diff_true = y_true_safe.unsqueeze(2) - y_true_safe.unsqueeze(1)
#
#     # Use the sign function instead of sigmoid
#     pairwise_score = torch.sign(diff_pred * diff_true) * pairwise_mask.float()
#
#     num_pairs = pairwise_mask.sum(dim=(1,2)).float()
#     valid_len = mask.sum(dim=1)  # (N,)
#     valid_row = valid_len >= nvv
#
#     row_losses = torch.where(
#         valid_row,
#         pairwise_score.sum(dim=(1,2)) / (num_pairs + eps),
#         torch.tensor(float('nan'), device=y_pred.device)
#     )
#
#     loss_values = [trimmed_mean(row_losses, trim=trim)]
#     loss = 1 - loss_values[0]
#     return loss

import torch

# def pairwise_monotone_loss_rowwise_vec_safe_tanh_dual_scale(
#         y_pred,
#         y_true,
#         trim=0.3,
#         nvv=10,
#         tau=0.1,
#         eps=1e-6
# ):
#     """
#     Strict pairwise monotone loss that completely ignores NaNs:
#     - Only non-NaN values participate in all computations
#     - Basin-wise normalization is computed on non-NaN values
#     Input shape: (T, N, 1)
#     """
#
#     # --------------------------------------------------
#     # 0. reshape to (N, T)
#     # --------------------------------------------------
#     y_pred = y_pred.squeeze(-1).permute(1, 0)  # (N, T)
#     y_true = y_true.squeeze(-1).permute(1, 0)  # (N, T)
#
#     N, T = y_true.shape
#
#     # --------------------------------------------------
#     # 1. mask
#     # --------------------------------------------------
#     mask = (~torch.isnan(y_true)) & (~torch.isnan(y_pred))  # (N, T)
#
#     # --------------------------------------------------
#     # 2. basin-wise scales (only non-NaN)
#     # --------------------------------------------------
#     scale_true = torch.zeros(N, 1, device=y_true.device)
#     scale_pred = torch.zeros(N, 1, device=y_pred.device)
#     for i in range(N):
#         valid_idx = mask[i]
#         if valid_idx.sum() > 0:
#             scale_true[i] = y_true[i, valid_idx].std(unbiased=False) + eps
#             scale_pred[i] = y_pred[i, valid_idx].std(unbiased=False) + eps
#         else:
#             scale_true[i] = torch.nan
#             scale_pred[i] = torch.nan
#
#     # --------------------------------------------------
#     # 3. normalized valid values
#     # --------------------------------------------------
#     y_true_norm = torch.full_like(y_true, float('nan'))
#     y_pred_norm = torch.full_like(y_pred, float('nan'))
#     for i in range(N):
#         valid_idx = mask[i]
#         if valid_idx.sum() > 0:
#             y_true_norm[i, valid_idx] = y_true[i, valid_idx] / scale_true[i]
#             y_pred_norm[i, valid_idx] = y_pred[i, valid_idx] / scale_pred[i]
#
#     # --------------------------------------------------
#     # 4. row-wise pairwise score
#     # --------------------------------------------------
#     row_losses = torch.full((N,), float('nan'), device=y_true.device)
#     for i in range(N):
#         valid_idx=torch.nonzero(mask[i]).squeeze(-1)
#         n_valid = len(valid_idx)
#         if n_valid >= nvv:
#             # pairwise differences
#             t_true = y_true_norm[i, valid_idx]
#             t_pred = y_pred_norm[i, valid_idx]
#             diff_true = t_true.unsqueeze(1) - t_true.unsqueeze(0)
#             diff_pred = t_pred.unsqueeze(1) - t_pred.unsqueeze(0)
#             pairwise_score = torch.tanh(diff_true * diff_pred / tau)
#             row_losses[i] = pairwise_score.mean()
#
#     # --------------------------------------------------
#     # 5. trimmed mean over rows
#     # --------------------------------------------------
#     valid_rows = row_losses[~torch.isnan(row_losses)]
#     if len(valid_rows) == 0:
#         loss_value = torch.tensor(float('nan'), device=y_true.device)
#     else:
#         sorted_rows, _ = torch.sort(valid_rows)
#         n = len(sorted_rows)
#         k = int(n * trim)
#         if n > 2 * k:
#             loss_value = sorted_rows[k:n-k].mean()
#         else:
#             loss_value = sorted_rows.mean()
#
#     loss = 1.0 - loss_value
#     return loss

# def pairwise_monotone_loss_rowwise_vec_safe_tanh_dual_scale_left_trim(
#         y_pred,
#         y_true,
#         trim=0.3,
#         nvv=10,
#         tau=0.1,
#         eps=1e-6
# ):
#     """
#     Strict pairwise monotone loss that completely ignores NaNs:
#     - Only non-NaN values participate in all computations
#     - Basin-wise normalization is computed on non-NaN values
#     Input shape: (T, N, 1)
#     """
#
#     # --------------------------------------------------
#     # 0. reshape to (N, T)
#     # --------------------------------------------------
#     y_pred = y_pred.squeeze(-1).permute(1, 0)  # (N, T)
#     y_true = y_true.squeeze(-1).permute(1, 0)  # (N, T)
#
#     N, T = y_true.shape
#
#     # --------------------------------------------------
#     # 1. mask
#     # --------------------------------------------------
#     mask = (~torch.isnan(y_true)) & (~torch.isnan(y_pred))  # (N, T)
#
#     # --------------------------------------------------
#     # 2. basin-wise scales (only non-NaN)
#     # --------------------------------------------------
#     scale_true = torch.zeros(N, 1, device=y_true.device)
#     scale_pred = torch.zeros(N, 1, device=y_pred.device)
#     for i in range(N):
#         valid_idx = mask[i]
#         if valid_idx.sum() > 0:
#             scale_true[i] = y_true[i, valid_idx].std(unbiased=False) + eps
#             scale_pred[i] = y_pred[i, valid_idx].std(unbiased=False) + eps
#         else:
#             scale_true[i] = torch.nan
#             scale_pred[i] = torch.nan
#
#     # --------------------------------------------------
#     # 3. normalized valid values
#     # --------------------------------------------------
#     y_true_norm = torch.full_like(y_true, float('nan'))
#     y_pred_norm = torch.full_like(y_pred, float('nan'))
#     for i in range(N):
#         valid_idx = mask[i]
#         if valid_idx.sum() > 0:
#             y_true_norm[i, valid_idx] = y_true[i, valid_idx] / scale_true[i]
#             y_pred_norm[i, valid_idx] = y_pred[i, valid_idx] / scale_pred[i]
#
#     # --------------------------------------------------
#     # 4. row-wise pairwise score
#     # --------------------------------------------------
#     row_losses = torch.full((N,), float('nan'), device=y_true.device)
#     for i in range(N):
#         valid_idx=torch.nonzero(mask[i]).squeeze(-1)
#         n_valid = len(valid_idx)
#         if n_valid >= nvv:
#             # pairwise differences
#             t_true = y_true_norm[i, valid_idx]
#             t_pred = y_pred_norm[i, valid_idx]
#             diff_true = t_true.unsqueeze(1) - t_true.unsqueeze(0)
#             diff_pred = t_pred.unsqueeze(1) - t_pred.unsqueeze(0)
#             pairwise_score = torch.tanh(diff_true * diff_pred / tau)
#             row_losses[i] = pairwise_score.mean()
#
#     # --------------------------------------------------
#     # 5. trimmed mean over rows
#     # --------------------------------------------------
#     valid_rows = row_losses[~torch.isnan(row_losses)]
#     if len(valid_rows) == 0:
#         loss_value = torch.tensor(float('nan'), device=y_true.device)
#     else:
#         sorted_rows, _ = torch.sort(valid_rows)
#         n = len(sorted_rows)
#         k = int(n * trim)
#         # Original code:
#         # if n > 2 * k:
#         #     loss_value = sorted_rows[k:n-k].mean()
#         # else:
#         #     loss_value = sorted_rows.mean()
#
#         # Modified: remove only the lower tail
#         if n>k:
#             loss_value=sorted_rows[k:].mean()  # Take from index k through the last element
#         else:
#             loss_value=sorted_rows.mean()
#
#     loss = 1.0 - loss_value
#     return loss

def pairwise_monotone_loss_rowwise_vec_safe_tanh_dual_scale_left_trim(
        y_pred,
        y_true,
        trim=0.3,
        nvv=10,
        tau=0.1,
        eps=1e-6
):
    # 1. Reshape
    # Ensure the data are converted to floating point before conversion, just in case
    y_pred=y_pred.squeeze(-1).transpose(0,1).float()  # (N, T)
    y_true=y_true.squeeze(-1).transpose(0,1).float()  # (N, T)

    # 2. Masking
    # Keep only finite values (also mask NaN/Inf) to prevent propagation of invalid values
    mask=torch.isfinite(y_true)&torch.isfinite(y_pred)
    # Convert to FloatTensor for subsequent weighted statistics
    mask_float=mask.float()

    y_true_clean=torch.where(mask,y_true,torch.tensor(0.0,device=y_true.device))
    y_pred_clean=torch.where(mask,y_pred,torch.tensor(0.0,device=y_pred.device))

    # 3. Vectorized Standard Deviation
    # Key fix: ensure counts are floating-point
    n_valid_per_row=mask_float.sum(dim=1,keepdim=True)

    def get_std(x,counts):
        # Mean computation
        mean=x.sum(dim=1,keepdim=True)/(counts+eps)
        # Key fix: use mask_float (FloatTensor) for the computation
        var=((x-mean)**2*mask_float).sum(dim=1,keepdim=True)/(counts+eps)
        return torch.sqrt(torch.clamp(var,min=0.0))+eps

    scale_true=get_std(y_true_clean,n_valid_per_row)
    scale_pred=get_std(y_pred_clean,n_valid_per_row)

    # 4. Normalization
    y_t_norm=y_true_clean/scale_true
    y_p_norm=y_pred_clean/scale_pred

    # 5. Vectorized Pairwise Score
    # Generate an (N, T, T) matrix
    diff_true=y_t_norm.unsqueeze(2)-y_t_norm.unsqueeze(1)
    diff_pred=y_p_norm.unsqueeze(2)-y_p_norm.unsqueeze(1)

    # Pairwise mask: bool for masked_fill, float for pair counting
    pair_mask=mask.unsqueeze(2)&mask.unsqueeze(1)
    pair_mask_float=pair_mask.float()

    # Clip the input first, then apply tanh and explicitly replace any NaN/Inf
    # Purpose: prevent extreme or invalid values from producing non-finite results
    pairwise_input=torch.clamp((diff_true*diff_pred)/tau,min=-30.0,max=30.0)
    pairwise_matrix=torch.tanh(pairwise_input)
    # Compatible with older PyTorch (torch does not have nan_to_num in the mhpihydrodl environment)
    # tanh outputs values in [-1,1], so only non-finite values need to be replaced with 0 here.
    if hasattr(torch, 'nan_to_num'):
        pairwise_matrix=torch.nan_to_num(pairwise_matrix,nan=0.0,posinf=1.0,neginf=-1.0)
    else:
        pairwise_matrix=torch.where(torch.isfinite(pairwise_matrix),pairwise_matrix,torch.zeros_like(pairwise_matrix))
    # Do not use a multiplicative mask (NaN*0 can still be NaN); use masked_fill for greater stability
    pairwise_matrix=pairwise_matrix.masked_fill(~pair_mask,0.0)

    # Average only over valid pairs
    sum_scores=pairwise_matrix.sum(dim=(1,2))
    n_pairs=pair_mask_float.sum(dim=(1,2))

    row_losses=sum_scores/(n_pairs+eps)

    # 6. Filter and Trimmed Mean
    # Also require enough valid samples and a finite row loss
    valid_row_mask=(n_valid_per_row.squeeze()>=nvv)&torch.isfinite(row_losses)
    valid_rows=row_losses[valid_row_mask]

    if valid_rows.numel()==0:
        return torch.tensor(0.0,device=y_true.device,requires_grad=True)

    sorted_rows,_=torch.sort(valid_rows)
    n=sorted_rows.numel()
    k=int(n*trim)

    loss_value=sorted_rows[k:].mean() if n>k else sorted_rows.mean()
    loss = 1.0 - loss_value
    return loss

# def pairwise_monotone_loss_rowwise_vec_safe_tanh_dual_scale_zoero(
#         y_pred,
#         y_true,
#         trim=0.3,
#         nvv=10,
#         tau=0.1,
#         eps=1e-6
# ):
#     """
#     Pairwise monotone loss with:
#     - basin-wise scale for y_true and y_pred (separately)
#     - tanh smooth surrogate
#     Compatible with PyTorch versions without torch.nanstd
#     """
#
#     # --------------------------------------------------
#     # 0. reshape to (N, T)
#     # --------------------------------------------------
#     y_pred = y_pred.squeeze(-1).permute(1, 0)  # (N, T)
#     y_true = y_true.squeeze(-1).permute(1, 0)  # (N, T)
#
#
#
#     # --------------------------------------------------
#     # 1. mask
#     # --------------------------------------------------
#     mask = (~torch.isnan(y_pred)) & (~torch.isnan(y_true))  # (N, T)
#     pairwise_mask = mask.unsqueeze(2) & mask.unsqueeze(1)   # (N, T, T)
#
#     # --------------------------------------------------
#     # 2. basin-wise scales (SEPARATELY, compatible nanstd)
#     # --------------------------------------------------
#     def nanstd(x, eps=1e-6):
#         mask_x = ~torch.isnan(x)
#         count_valid = mask_x.sum(dim=1, keepdim=True).float()       # (N,1) <- cast to float
#         sum_x = torch.where(mask_x, x, torch.zeros_like(x)).sum(dim=1, keepdim=True)
#         mean_x = sum_x / count_valid
#         var_x = torch.where(mask_x, (x - mean_x)**2, torch.zeros_like(x)).sum(dim=1, keepdim=True) / count_valid
#         return torch.sqrt(var_x + eps)
#
#     scale_true = nanstd(y_true, eps=eps)
#     scale_pred = nanstd(y_pred, eps=eps)
#
#     # --------------------------------------------------
#     # 3. safe fill
#     # --------------------------------------------------
#     y_true_safe = torch.where(mask, y_true, torch.zeros_like(y_true))
#     y_pred_safe = torch.where(mask, y_pred, torch.zeros_like(y_pred))
#
#     # --------------------------------------------------
#     # 4. normalize separately
#     # --------------------------------------------------
#     y_true_norm = y_true_safe / scale_true
#     y_pred_norm = y_pred_safe / scale_pred
#
#     # --------------------------------------------------
#     # 5. pairwise differences
#     # --------------------------------------------------
#     diff_true = y_true_norm.unsqueeze(2) - y_true_norm.unsqueeze(1)
#     diff_pred = y_pred_norm.unsqueeze(2) - y_pred_norm.unsqueeze(1)
#
#     # --------------------------------------------------
#     # 6. tanh-based monotonic score
#     # --------------------------------------------------
#     pairwise_score = torch.tanh((diff_pred * diff_true) / tau) * pairwise_mask.float()
#
#     # --------------------------------------------------
#     # 7. row-wise aggregation
#     # --------------------------------------------------
#     num_pairs = pairwise_mask.sum(dim=(1, 2)).float()
#     valid_len = mask.sum(dim=1)
#     valid_row = valid_len >= nvv
#
#     row_losses = torch.where(
#         valid_row,
#         pairwise_score.sum(dim=(1, 2)) / (num_pairs + eps),
#         torch.tensor(float('nan'), device=y_pred.device)
#     )
#
#     # --------------------------------------------------
#     # 8. trimmed mean
#     # --------------------------------------------------
#     loss_value = trimmed_mean(row_losses, trim=trim)
#     loss = 1.0 - loss_value
#     return loss




class NSELossBatch_li_multi(torch.nn.Module):
    # Same as Fredrick 2019, batch NSE loss
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray, alpha =1, nvv= 10,eps=0.1):
        super(NSELossBatch_li_multi, self).__init__()
        self.std = stdarray
        self.eps = eps
        self.alpha = alpha
        self.nvv = nvv

    def forward(self, output, target, igrid):
        if len(np.shape(output))>2:
            nt = target.shape[0]
            stdse = np.tile(self.std[igrid][None,:], (nt, 1,1))
            stdbatch = torch.tensor(stdse, requires_grad=False).float().to(output)
        # else:
        #     stdse=np.tile(self.std[igrid][None,:],(nt,1,1))
        #     stdbatch=torch.tensor(stdse,requires_grad=False).float().to(output)
        if len(np.shape(output))<3:
            p0=output[:,:target.shape[1]]  # dim: Time*Gage
            t0=target[:,:]
        else:
            p0 = output[:, :, :]   # dim: Time*Gage
            t01 = target[:, :, 0, None]
            t02 = target[:, :, 1, None]
        mask1 = t01 == t01
        p1 = p0[mask1]
        t1 = t01[mask1]
        stdw = stdbatch[mask1]
        sqRes = (p1 - t1)**2
        normRes = sqRes / (stdw + self.eps)**2
        loss = torch.mean(normRes)

        loss2 = corr_loss(p0,t02, self.nvv)

        # if t2.numel()>0:
        #     print("Tensor is not empty.")
        # sqRes = (t0 - p0)**2 # squared error
        # normRes = sqRes / (stdbatch + self.eps)**2
        # mask = t0 == t0
        # loss = torch.mean(normRes[mask])
        # print(loss, loss2)
        # print(loss.cpu().detach().numpy())
        # print(loss2.cpu().detach().numpy())
        return loss+loss2 * self.alpha


# def corr_loss_torch(y_obs, y_sim):
#     mask = ~torch.isnan(y_obs) & ~torch.isnan(y_sim)
#     y_obs, y_sim = y_obs[mask], y_sim[mask]
#     if y_obs.numel() < 2:
#         return torch.tensor(0.0)  # or other handling
#     y_obs_mean = torch.mean(y_obs)
#     y_sim_mean = torch.mean(y_sim)
#     cov = torch.sum((y_obs - y_obs_mean)*(y_sim - y_sim_mean))
#     stds = torch.sqrt(torch.sum((y_obs - y_obs_mean)**2) * torch.sum((y_sim - y_sim_mean)**2))
#     r = cov / (stds + 1e-8)
#     return 1 - r

class NSELossBatch_li_multi_space(torch.nn.Module):
    # Same as Fredrick 2019, batch NSE loss
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray, alpha =1, nvv= 10,trim = 0.3,eps=0.1,eps_wse=0.3):
        super(NSELossBatch_li_multi_space, self).__init__()
        self.std = stdarray
        self.eps = eps
        self.alpha = alpha
        self.nvv = nvv
        self.trim = trim
        self.eps_wse = eps_wse

    def forward(self, output, target, idx, igrid, batchSize):
        if len(np.shape(output))>2:
            nt = target.shape[0]
            stdse = np.tile(self.std[idx][igrid[:batchSize]][None,:], (nt, 1,1))
            stdbatch = torch.tensor(stdse, requires_grad=False).float().to(output)
        # else:
        #     stdse=np.tile(self.std[igrid][None,:],(nt,1,1))
        #     stdbatch=torch.tensor(stdse,requires_grad=False).float().to(output)
        if len(np.shape(output))<3:
            p0=output[:,:target.shape[1]]  # dim: Time*Gage
            t0=target[:,:]
        else:
            p01 = output[:, :batchSize, :]   # dim: Time*Gage
            p02 = output[:, batchSize:, :]
            t01 = target[:, :batchSize, 0, None]
            t02 = target[:, batchSize:, 1, None]
        mask1 = t01 == t01
        p1 = p01[mask1]
        t1 = t01[mask1]
        stdw = stdbatch[mask1]
        sqRes = (p1 - t1)**2
        normRes = sqRes / (stdw + self.eps)**2
        loss = torch.mean(normRes)

        # mask2=t02==t02
        # p2 = p02[mask2]
        # t2 = t02[mask2]
        # loss2 = corr_loss_torch(t2, p2)
        # loss2 = corr_loss(p02, t02, self.nvv)
        loss2 = corr_loss_trim(p02,t02,self.nvv, self.trim)
        loss2=torch.clamp(loss2,min=self.eps_wse)


        # if t2.numel()>0:
        #     print("Tensor is not empty.")
        # sqRes = (t0 - p0)**2 # squared error
        # normRes = sqRes / (stdbatch + self.eps)**2
        # mask = t0 == t0
        # loss = torch.mean(normRes[mask])
        # print(loss, loss2)
        # print(loss.cpu().detach().numpy())
        # print(loss2.cpu().detach().numpy())
        if abs(self.alpha - 0.0) <1e-4:
            return loss, loss2
        return loss+loss2 * self.alpha, loss2


class NSELossBatch_li_multi_space_separate(torch.nn.Module):
    # Same as Fredrick 2019, batch NSE loss
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray, alpha =1, nvv= 10,trim = 0.3,eps=0.1,clamp_eps=0.3,clamp_if= False, kind = 0):
        super(NSELossBatch_li_multi_space_separate, self).__init__()
        self.std = stdarray
        self.eps = eps
        self.alpha = alpha
        self.nvv = nvv
        self.trim = trim
        self.eps_wse = clamp_eps
        self.clamp_if = clamp_if
        self.kind = kind

    def forward(self, output, output2, target, target2, idx, igrid, batchSize):
        if len(np.shape(output))>2:
            nt = target.shape[0]
            stdse = np.tile(self.std[idx][igrid[:batchSize]][None,:], (nt, 1,1))
            stdbatch = torch.tensor(stdse, requires_grad=False).float().to(output)
        # else:
        #     stdse=np.tile(self.std[igrid][None,:],(nt,1,1))
        #     stdbatch=torch.tensor(stdse,requires_grad=False).float().to(output)
        if len(np.shape(output))<3:
            p0=output[:,:target.shape[1]]  # dim: Time*Gage
            t0=target[:,:]
        else:
            p01 = output[:, :, :]   # dim: Time*Gage
            p02 = output2[:,:, :]
            t01 = target[:, :, 0, None]
            t02 = target2[:, :, 1, None]
        mask1 = t01 == t01
        p1 = p01[mask1]
        t1 = t01[mask1]
        stdw = stdbatch[mask1]
        sqRes = (p1 - t1)**2
        normRes = sqRes / (stdw + self.eps)**2
        loss = torch.mean(normRes)

        # mask2=t02==t02
        # p2 = p02[mask2]
        # t2 = t02[mask2]
        # loss2 = corr_loss_torch(t2, p2)
        # loss2 = corr_loss(p02, t02, self.nvv)
        if self.kind == 0:
            loss2 = corr_loss_trim(p02,t02,self.nvv, self.trim)
        elif self.kind == 1:
            loss2 = soft_spearman_loss_parallel_nopad(p02, t02, trim = self.trim, nvv = self.nvv)
        elif self.kind == 2:
            loss2 = pairwise_monotone_loss_rowwise_vec_safe_tanh_dual_scale(p02,t02,trim = self.trim, nvv = self.nvv)
        if self.clamp_if == True:
            loss2=torch.clamp(loss2,min=self.eps_wse)


        # if t2.numel()>0:
        #     print("Tensor is not empty.")
        # sqRes = (t0 - p0)**2 # squared error
        # normRes = sqRes / (stdbatch + self.eps)**2
        # mask = t0 == t0
        # loss = torch.mean(normRes[mask])
        # print(loss, loss2)
        # print(loss.cpu().detach().numpy())
        # print(loss2.cpu().detach().numpy())
        if abs(self.alpha - 0.0) <1e-4:
            return loss, loss2
        return loss+loss2 * self.alpha, loss2

class NSELossBatch_li_multi_space_separate_swot_hydrosheds(torch.nn.Module):
    # Same as Fredrick 2019, batch NSE loss
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray, alpha =1, nvv= 10,trim = 0.3,eps=0.1,clamp_eps=0.3,clamp_if= False, kind = 0):
        super(NSELossBatch_li_multi_space_separate_swot_hydrosheds, self).__init__()
        self.std = stdarray
        self.eps = eps
        self.alpha = alpha
        self.nvv = nvv
        self.trim = trim
        self.eps_wse = clamp_eps
        self.clamp_if = clamp_if
        self.kind = kind

    def forward(self, output, output2, target, target2, idx, igrid, batchSize):
        if len(np.shape(output))>2:
            nt = target.shape[0]
            stdse = np.tile(self.std[idx][igrid[:batchSize]][None,:], (nt, 1,1))
            stdbatch = torch.tensor(stdse, requires_grad=False).float().to(output)
        # else:
        #     stdse=np.tile(self.std[igrid][None,:],(nt,1,1))
        #     stdbatch=torch.tensor(stdse,requires_grad=False).float().to(output)
        if len(np.shape(output))<3:
            p0=output[:,:target.shape[1]]  # dim: Time*Gage
            t0=target[:,:]
        else:
            p01 = output[:, :, :]   # dim: Time*Gage
            p02 = output2[:,:, :]
            t01 = target[:, :, :]
            t02 = target2[:, :, :]
        # Keep only finite values: target, output, and std must all be finite
        mask1 = torch.isfinite(t01)&torch.isfinite(p01)&torch.isfinite(stdbatch)
        p1 = p01[mask1]
        t1 = t01[mask1]
        stdw = stdbatch[mask1]
        if p1.numel()==0:
            # Return 0 when there are no valid samples to avoid NaN from the mean of an empty tensor.
            loss = torch.tensor(0.0,device=output.device,requires_grad=True)
        else:
            sqRes = (p1 - t1)**2
            # Protect the denominator lower bound to avoid numerical explosion when (std+eps)^2 is too small
            den = torch.clamp((stdw + self.eps)**2,min=self.eps**2)
            normRes = sqRes / den
            loss = torch.mean(normRes)
            if not bool(torch.isfinite(loss).item()):
                # Secondary safeguard: fall back to 0 if a non-finite value occurs to avoid interrupting training
                loss = torch.tensor(0.0,device=output.device,requires_grad=True)

        # mask2=t02==t02
        # p2 = p02[mask2]
        # t2 = t02[mask2]
        # loss2 = corr_loss_torch(t2, p2)
        # loss2 = corr_loss(p02, t02, self.nvv)
        if abs(self.alpha - 0.0) <1e-4:
            return loss, loss

        if self.kind == 0:
            loss2 = corr_loss_trim_fast(p02,t02,self.nvv, self.trim)
        elif self.kind == 1:
            loss2 = soft_spearman_loss_parallel_nopad(p02, t02, trim = self.trim, nvv = self.nvv)
        elif self.kind == 2:
            # loss2 = pairwise_monotone_loss_rowwise_vec_safe_tanh_dual_scale(p02,t02,trim = self.trim, nvv = self.nvv)
            loss2=pairwise_monotone_loss_rowwise_vec_safe_tanh_dual_scale_left_trim(p02,t02,trim=self.trim,nvv=self.nvv)
        if not bool(torch.isfinite(loss2).item()):
            # Fall back to 0 when the correlation term is non-finite so training can continue with the main loss
            loss2=torch.tensor(0.0,device=output.device,requires_grad=True)
        if self.clamp_if == True:
            loss2=torch.clamp(loss2,min=self.eps_wse)


        # if t2.numel()>0:
        #     print("Tensor is not empty.")
        # sqRes = (t0 - p0)**2 # squared error
        # normRes = sqRes / (stdbatch + self.eps)**2
        # mask = t0 == t0
        # loss = torch.mean(normRes[mask])
        # print(loss, loss2)
        # print(loss.cpu().detach().numpy())
        # print(loss2.cpu().detach().numpy())
        return loss+loss2 * self.alpha, loss2

#
# class NSELossBatch_li_smooth_lake(torch.nn.Module):
#     # Same as Fredrick 2019, batch NSE loss
#     # stdarray: the standard deviation of the runoff for all basins
#     def __init__(self, stdarray, eps=0.1, lambda_smooth=0.005):
#         super(NSELossBatch_li_smooth_lake, self).__init__()
#         self.std = stdarray
#         self.eps = eps
#         self.lambda_smooth = lambda_smooth
#
#     def forward(self, output, target, igrid):
#         nt = target.shape[0]
#         stdse = np.tile(self.std[igrid][None,:], (nt, 1,1))
#         stdbatch = torch.tensor(stdse, requires_grad=False).float().to(output)
#         p0 = output[:, :, :]   # dim: Time*Gage
#         t0 = target[:, :, :]
#         mask = t0 == t0
#         p = p0[mask]
#         t = t0[mask]
#         stdw = stdbatch[mask]
#         sqRes = (p - t)**2
#         normRes = sqRes / (stdw + self.eps)**2
#         loss = torch.mean(normRes)
#
#         # Compute the smoothness penalty term
#         pb = p0[1:]
#         pc = p0[:-1]
#
#         p0_s=output[:,:,:target.shape[2]]  # dim: Time*Gage
#         pb_s = p0[1:]
#         pc_s = p0[:-1]
#         # smoothness_penalty=torch.sum(torch.abs(pb.view(-1)-pc.view(-1)))
#         smoothness_penalty=torch.sum((pb.view(-1)-pc.view(-1))**2)
#         A1 = torch.sum((pb.view(-1)-pc.view(-1))**2)
#         A2 = torch.sum((pb_s.view(-1)-pc_s.view(-1))**2)
#         A3 = torch.mean((pb.view(-1)-pc.view(-1))**2)
#         A4 = self.lambda_smooth*smoothness_penalty/A3
#         print(A1/A2)
#         # Apply smoothness penalty only if predicted values are below 0.5
#         # condition = torch.abs(pc-pb) < 0.5
#         # smoothness_penalty = torch.sum(torch.abs(pb.view(-1)-pc.view(-1)) * condition.view(-1).float())
#
#         # Total loss = original loss + smoothness penalty
#         total_loss=loss+self.lambda_smooth*smoothness_penalty
#
#         # sqRes = (t0 - p0)**2 # squared error
#         # normRes = sqRes / (stdbatch + self.eps)**2
#         # mask = t0 == t0
#         # loss = torch.mean(normRes[mask])
#         return total_loss

# class NSELossBatch_li_smooth(torch.nn.Module):
#     # Same as Fredrick 2019, batch NSE loss
#     # stdarray: the standard deviation of the runoff for all basins
#     def __init__(self, stdarray, eps=0.1, lambda_smooth=0.005):
#         super(NSELossBatch_li_smooth, self).__init__()
#         self.std = stdarray
#         self.eps = eps
#         self.lambda_smooth = lambda_smooth
#
#     def forward(self, output, target, igrid):
#         nt = target.shape[0]
#         stdse = np.tile(self.std[igrid][None,:], (nt, 1,1))
#         stdbatch = torch.tensor(stdse, requires_grad=False).float().to(output)
#         p0 = output[:, :, :target.shape[2]]   # dim: Time*Gage
#         t0 = target[:, :, :]
#         mask = t0 == t0
#         p = p0[mask]
#         t = t0[mask]
#         stdw = stdbatch[mask]
#         sqRes = (p - t)**2
#         normRes = sqRes / (stdw + self.eps)**2
#         loss = torch.mean(normRes)
#
#         # Compute the smoothness penalty term
#         pb = p0[1:]
#         pc = p0[:-1]
#         # smoothness_penalty=torch.sum(torch.abs(pb.view(-1)-pc.view(-1)))
#         smoothness_penalty=torch.sum((pb.view(-1)-pc.view(-1))**2)
#         # Apply smoothness penalty only if predicted values are below 0.5
#         # condition = torch.abs(pc-pb) < 0.5
#         # smoothness_penalty = torch.sum(torch.abs(pb.view(-1)-pc.view(-1)) * condition.view(-1).float())
#         # smoothness_penalty=torch.sum((pb.view(-1)-pc.view(-1))**2*condition.view(-1).float())
#
#         # Total loss = original loss + smoothness penalty
#         total_loss=loss+self.lambda_smooth*smoothness_penalty
#
#         # sqRes = (t0 - p0)**2 # squared error
#         # normRes = sqRes / (stdbatch + self.eps)**2
#         # mask = t0 == t0
#         # loss = torch.mean(normRes[mask])
#         return total_loss

class NSELossBatch_li_smooth(torch.nn.Module):
    # Same as Fredrick 2019, batch NSE loss
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray, eps=0.1, lambda_smooth=1.8):
        # change roughness penalty lamda from 0.005 to 1.8
        # change the torch.sum to torch.mean for roughness penalty
        super(NSELossBatch_li_smooth, self).__init__()
        self.std = stdarray
        self.eps = eps
        self.lambda_smooth = lambda_smooth

    def forward(self, output, target, igrid):
        nt = target.shape[0]
        stdse = np.tile(self.std[igrid][None,:], (nt, 1,1))
        stdbatch = torch.tensor(stdse, requires_grad=False).float().to(output)
        p0 = output[:, :, :target.shape[2]]   # dim: Time*Gage
        t0 = target[:, :, :]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        stdw = stdbatch[mask]
        sqRes = (p - t)**2
        normRes = sqRes / (stdw + self.eps)**2
        loss = torch.mean(normRes)

        # Compute the smoothness penalty term
        pb = p0[1:]
        pc = p0[:-1]
        # smoothness_penalty=torch.sum(torch.abs(pb.view(-1)-pc.view(-1)))
        smoothness_penalty=torch.mean((pb.view(-1)-pc.view(-1))**2)
        # Apply smoothness penalty only if predicted values are below 0.5
        # condition = torch.abs(pc-pb) < 0.5
        # smoothness_penalty = torch.sum(torch.abs(pb.view(-1)-pc.view(-1)) * condition.view(-1).float())
        # smoothness_penalty=torch.sum((pb.view(-1)-pc.view(-1))**2*condition.view(-1).float())

        A1=torch.mean((pb.view(-1)-pc.view(-1))**2)
        A2=torch.sum((pb.view(-1)-pc.view(-1))**2)
        A3=torch.mean((pb.view(-1)-pc.view(-1))**2*110)
        # print(A1)
        # print(A2)
        print(A2*0.005/A1)
        # print((A2*0.005)/(A1*110))
        # print(A1*110-A3)



        # Total loss = original loss + smoothness penalty
        total_loss=loss+self.lambda_smooth*smoothness_penalty

        # sqRes = (t0 - p0)**2 # squared error
        # normRes = sqRes / (stdbatch + self.eps)**2
        # mask = t0 == t0
        # loss = torch.mean(normRes[mask])
        return total_loss


class NSESqrtLossBatch(torch.nn.Module):
    # Same as Fredrick 2019, batch NSE loss, use RMSE and STD instead
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray, eps=0.1):
        super(NSESqrtLossBatch, self).__init__()
        self.std = stdarray
        self.eps = eps

    def forward(self, output, target, igrid):
        nt = target.shape[0]
        stdse = np.tile(self.std[igrid], (nt, 1))
        stdbatch = torch.tensor(stdse, requires_grad=False).float().cuda()
        p0 = output[:, :, 0]   # dim: Time*Gage
        t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        stdw = stdbatch[mask]
        sqRes = torch.sqrt((p - t)**2)
        normRes = sqRes / (stdw + self.eps)
        loss = torch.mean(normRes)

        # sqRes = (t0 - p0)**2 # squared error
        # normRes = sqRes / (stdbatch + self.eps)**2
        # mask = t0 == t0
        # loss = torch.mean(normRes[mask])
        return loss

class TrendLoss(torch.nn.Module):
    # Add the trend part to the loss
    def __init__(self):
        super(TrendLoss, self).__init__()

    def getSlope(self, x):
        idx = 0
        n = len(x)
        d = torch.ones(int(n * (n - 1) / 2))

        for i in range(n - 1):
            j = torch.arange(start=i + 1, end=n)
            d[idx: idx + len(j)] = (x[j] - x[i]) / (j - i).type(torch.float)
            idx = idx + len(j)

        return torch.median(d)

    def forward(self, output, target, PercentLst=[100, 98, 50, 30, 2]):
        # output, target: rho/time * Batchsize * Ntraget_var
        ny = target.shape[2]
        nt = target.shape[0]
        ngage = target.shape[1]
        loss = 0
        for k in range(ny):
        # loop for variable
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            # first part loss, regular RMSE
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp
            temptrendloss = 0
            nsample = 0
            for ig in range(ngage):
                # loop for basins
                pgage0 = p0[:, ig].reshape(-1, 365)
                tgage0 = t0[:, ig].reshape(-1, 365)
                gBool = np.zeros(tgage0.shape[0]).astype(int)
                pgageM = torch.zeros(tgage0.shape[0])
                pgageQ = torch.zeros(tgage0.shape[0], len(PercentLst))
                tgageM = torch.zeros(tgage0.shape[0])
                tgageQ = torch.zeros(tgage0.shape[0], len(PercentLst))
                for ii in range(tgage0.shape[0]):
                    pgage = pgage0[ii, :]
                    tgage = tgage0[ii, :]
                    maskg = tgage == tgage
                    # quality control
                    if maskg.sum() > (1-2/12)*365:
                        gBool[ii] = 1
                        pgage = pgage[maskg]
                        tgage = tgage[maskg]
                        pgageM[ii] = pgage.mean()
                        tgageM[ii] = tgage.mean()
                        for ip in range(len(PercentLst)):
                            k = math.ceil(PercentLst[ip] / 100 * 365)
                            # pgageQ[ii, ip] = torch.kthvalue(pgage, k)[0]
                            # tgageQ[ii, ip] = torch.kthvalue(tgage, k)[0]
                            pgageQ[ii, ip] = torch.sort(pgage)[0][k-1]
                            tgageQ[ii, ip] = torch.sort(tgage)[0][k-1]
                # Quality control
                if gBool.sum()>6:
                    nsample = nsample + 1
                    pgageM = pgageM[gBool]
                    tgageM = tgageM[gBool]
                    # mean annual trend loss
                    temptrendloss = temptrendloss + (self.getSlope(tgageM)-self.getSlope(pgageM))**2
                    pgageQ = pgageQ[gBool, :]
                    tgageQ = tgageQ[gBool, :]
                    # quantile trend loss
                    for ii in range(tgageQ.shape[1]):
                        temptrendloss = temptrendloss + (self.getSlope(tgageQ[:, ii])-self.getSlope(pgageQ[:, ii]))**2

            loss = loss + temptrendloss/nsample

        return loss


class ModifyTrend(torch.nn.Module):
    # Add the trend part to the loss
    def __init__(self):
        super(ModifyTrend, self).__init__()

    def getSlope(self, x):
        nyear, ngage = x.shape
        # define difference matirx
        x = x.transpose(0,1)
        xtemp = x.repeat(1, nyear)
        xi = xtemp.reshape([ngage, nyear, nyear])
        xj = xi.transpose(1,2)
        # define i,j matrix
        im = torch.arange(nyear).repeat(nyear).reshape(nyear,nyear).type(torch.float)
        im = im.unsqueeze(0).repeat([ngage, 1, 1])
        jm = im.transpose(1,2)
        delta = 1.0/(im - jm)
        delta = delta.cuda()
        # calculate the slope matrix
        slopeMat = (xi - xj)*delta
        rid, cid = np.triu_indices(nyear, k=1)
        slope = slopeMat[:, rid, cid]
        senslope = torch.median(slope, dim=-1)[0]

        return senslope


    def forward(self, output, target, PercentLst=[-1]):
        # output, target: rho/time * Batchsize * Ntraget_var
        # PercentLst = [100, 98, 50, 30, 2, -1]
        ny = target.shape[2]
        nt = target.shape[0]
        ngage = target.shape[1]
        # loop for variable
        p0 = output[:, :, 0]
        t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        # first part loss, regular RMSE
        # loss = torch.sqrt(((p - t)**2).mean())
        # loss = ((p - t) ** 2).mean()
        loss = 0
        temptrendloss = 0
        # second loss: adding trend
        p1 = p0.reshape(-1, 365, ngage)
        t1 = t0.reshape(-1, 365, ngage)
        for ip in range(len(PercentLst)):
            k = math.ceil(PercentLst[ip] / 100 * 365)
            # pQ = torch.kthvalue(p1, k, dim=1)[0]
            # tQ = torch.kthvalue(t1, k, dim=1)[0]
            # output: dim=Year*gage
            if PercentLst[ip]<0:
                pQ = torch.mean(p1, dim=1)
                tQ = torch.mean(t1, dim=1)
            else:
                pQ = torch.sort(p1, dim=1)[0][:, k - 1, :]
                tQ = torch.sort(t1, dim=1)[0][:, k - 1, :]
            # temptrendloss = temptrendloss + ((self.getSlope(pQ) - self.getSlope(tQ)) ** 2).mean()
            temptrendloss = temptrendloss + ((pQ - tQ) ** 2).mean()
        loss = loss + temptrendloss

        return loss


class ModifyTrend1(torch.nn.Module):
    # Add the trend part to the loss
    def __init__(self):
        super(ModifyTrend1, self).__init__()

    def getM(self, n):
        M = np.zeros([n**2, n])
        s0 = np.zeros([n**2, 1])
        for j in range (n):
            for i in range(n):
                k = j*n+i
                if i<j:
                    factor = 1/(j-i)
                    M[k, j] = factor
                    M[k, i] = -factor
                else:
                    s0[k] = np.nan
        sind = np.argwhere(~np.isnan(s0))
        return M, sind


    def forward(self, output, target, PercentLst=[100, 98, 50, 30, 2]):
        # PercentLst = [100, 98, 50, 30, 2]
        # output, target: rho/time * Batchsize * Ntraget_var
        output = output.permute(2, 0, 1)
        target = target.permute(2, 0, 1)
        ny = target.shape[2]
        nt = target.shape[0]
        ngage = target.shape[1]
        # loop for variable
        p0 = output[:, :, 0]
        t0 = target[:, :, 0]
        # mask = t0 == t0
        # p = p0[mask]
        # t = t0[mask]
        # first part loss, regular RMSE
        loss = 0.0
        # loss = torch.sqrt(((p - t)**2).mean())
        # loss = ((p - t) ** 2).mean()
        # second loss: adding trend
        p1 = p0.reshape(-1, 365, ngage)
        t1 = t0.reshape(-1, 365, ngage)
        nyear = p1.shape[0]
        nsample = p1.shape[-1]
        M, s0 = self.getM(nyear)
        Mtensor = torch.from_numpy(M).type(torch.float).cuda()
        for ip in range(len(PercentLst)):
            k = math.ceil(PercentLst[ip] / 100 * 365)
            # pQ = torch.kthvalue(p1, k, dim=1)[0]
            # tQ = torch.kthvalue(t1, k, dim=1)[0]
            # output: dim=Year*gage
            pQ = torch.sort(p1, dim=1)[0][:, k - 1, :]
            tQ = torch.sort(t1, dim=1)[0][:, k - 1, :]
            # pQ = p1[:, 100, :]
            # tQ = t1[:, 100, :]
            temptrenddiff = 0.0
            for ig in range(nsample):
                trenddiff = (torch.median(torch.mv(Mtensor, pQ[:, ig])[s0[:,0]]) - \
                            torch.median(torch.mv(Mtensor, tQ[:, ig])[s0[:,0]]))**2
                temptrenddiff = temptrenddiff + trenddiff
            temptrendloss = temptrenddiff/nsample
            loss = loss + temptrendloss
        return loss

# class ModifyTrend1(torch.nn.Module):
#     # Test MSE loss for each percentile
#     def __init__(self):
#         super(ModifyTrend1, self).__init__()
#
#     def getM(self, n):
#         M = np.zeros([n**2, n])
#         s0 = np.zeros([n**2, 1])
#         for j in range (n):
#             for i in range(n):
#                 k = j*n+i
#                 if i<j:
#                     factor = 1/(j-i)
#                     M[k, j] = factor
#                     M[k, i] = -factor
#                 else:
#                     s0[k] = np.nan
#         sind = np.argwhere(~np.isnan(s0))
#         return M, sind
#
#
#     def forward(self, output, target, PercentLst=[100, 98, 50, 30, 2]):
#         # PercentLst = [100, 98, 50, 30, 2]
#         # output, target: rho/time * Batchsize * Ntraget_var
#         output = output.permute(2, 0, 1)
#         target = target.permute(2, 0, 1)
#         ny = target.shape[2]
#         nt = target.shape[0]
#         ngage = target.shape[1]
#         # loop for variable
#         p0 = output[:, :, 0]
#         t0 = target[:, :, 0]
#         mask = t0 == t0
#         p = p0[mask]
#         t = t0[mask]
#         # first part loss, regular RMSE
#         # loss = 0.0
#         # loss = torch.sqrt(((p - t)**2).mean())
#         loss = ((p - t) ** 2).mean()
#         # second loss: adding trend
#         p1 = p0.reshape(-1, 365, ngage)
#         t1 = t0.reshape(-1, 365, ngage)
#         nyear = p1.shape[0]
#         nsample = p1.shape[-1]
#         M, s0 = self.getM(nyear)
#         Mtensor = torch.from_numpy(M).type(torch.float).cuda()
#         for ip in range(len(PercentLst)):
#             k = math.ceil(PercentLst[ip] / 100 * 365)
#             # output: dim=Year*gage
#             pQ = torch.sort(p1, dim=1)[0][:, k - 1, :]
#             tQ = torch.sort(t1, dim=1)[0][:, k - 1, :]
#             # calculate mse of each percentile flow
#             mask = tQ == tQ
#             ptemp = pQ[mask]
#             ttemp = tQ[mask]
#             temploss = ((ptemp - ttemp) ** 2).mean()
#             loss = loss + temploss
#         return loss
