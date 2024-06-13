import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class CrossExchangingLayer(nn.Module):
    def __init__(self, d_model, nhead, theta, skip_connection=False, use_quantile=False, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5,
                 batch_first=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CrossExchangingLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)

        # feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activatiion = activation

        self.theta = theta  # select the attention value
        self.skip_connection = skip_connection
        self.use_quantile = use_quantile

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x, attn_weight = self.multihead_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True)
        return self.dropout1(x), attn_weight

    # feedforward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout2(self.activatiion(self.linear1(x))))
        return self.dropout2(x)

    # cross-replace block
    def _cr_block(self, x1, x2, attn_weight1, attn_weight2):
        cls_weight1 = attn_weight1[:, 0, :]
        cls_weight2 = attn_weight2[:, 0, :]

        x1_mean = torch.mean(x1, dim=-2)
        x2_mean = torch.mean(x2, dim=-2)

        for i in range(cls_weight1.shape[0]):
            if self.use_quantile:
                theta1 = np.quantile(
                    cls_weight1[i][1:].detach().cpu().numpy(), self.theta)
                theta2 = np.quantile(
                    cls_weight2[i][1:].detach().cpu().numpy(), self.theta)
            else:
                theta1 = self.theta
                theta2 = self.theta

            # except the first token, namely [cls]
            for j in range(1, cls_weight1.shape[1]):
                if cls_weight1[i][j] < theta1:
                    x1[i][j] = x2_mean[i] + \
                               x1[i][j] if self.skip_connection else x2_mean[i]
                if cls_weight2[i][j] < theta2:
                    x2[i][j] = x1_mean[i] + \
                               x2[i][j] if self.skip_connection else x1_mean[i]
        return x1, x2

    def forward(self, src1, src2, AT_attn, AV_attn, replace=False, first_layer=False, src1_mask=None, src1_key_padding_mask=None, src2_mask=None,
                src2_key_padding_mask=None):
        x1 = src1
        x2 = src2

        if first_layer:
            attn_weight1 = AT_attn
            attn_weight2 = AV_attn
            if replace:
                x1, x2 = self._cr_block(x1, x2, attn_weight1, attn_weight2)
            x1 = x1 + self._ff_block(self.norm1(x1))
            x2 = x2 + self._ff_block(self.norm1(x2))

        else:
            res1, attn_weight1 = self._sa_block(x1, src1_mask, src1_key_padding_mask)
            res2, attn_weight2 = self._sa_block(x2, src2_mask, src2_key_padding_mask)
            x1 = self.norm1(x1 + res1)
            x2 = self.norm1(x2 + res2)
            if replace:
                x1, x2 = self._cr_block(x1, x2, attn_weight1, attn_weight2)
            x1 = self.norm2(x1 + self._ff_block(x1))
            x2 = self.norm2(x2 + self._ff_block(x2))
        return x1, x2

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CrossExchangingBlock(nn.Module):
    def __init__(self, encoder_layer, num_layers, replace_start, replace_end, norm=None):
        super(CrossExchangingBlock, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.replace_start = replace_start
        self.replace_end = replace_end

    def forward(self, src1, src2, AT_attn, AV_attn, mask1=None, src1_key_padding_mask=None, mask2=None, src2_key_padding_mask=None):
        output1 = src1
        output2 = src2
        for i, mod in enumerate(self.layers):
            if i >= self.replace_start and i <= self.replace_end:
                replace = True
            else:
                replace = False
            if i==0:
                first_layer = True
            else:
                first_layer = False
            output1, output2 = mod(output1, output2, AT_attn, AV_attn, replace=replace, first_layer=first_layer, src1_mask=mask1,
                                   src1_key_padding_mask=src1_key_padding_mask, src2_mask=mask2, src2_key_padding_mask=src2_key_padding_mask)
        if self.norm is not None:
            output1 = self.norm(output1)
            output2 = self.norm(output2)
        return output1, output2


