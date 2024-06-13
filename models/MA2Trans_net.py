import copy
from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from exhcanging import CrossExchangingLayer, CrossExchangingBlock
from visual_encoder import ObjEncoder
from attribute_encoder import Attributes
from transformers import BertConfig, BertModel

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class A_T_DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,):
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(
                query=tgt2, key=memory,
                value=memory, attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask)
        AT_attn = cross_attn_matrices
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, AT_attn

class A_V_DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,):
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask
            )
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(
                query=tgt2, key=memory,
                value=memory, attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices

def get_mlp_head(input_size, hidden_size, output_size, dropout=0):
    return nn.Sequential(
                nn.Linear(input_size, hidden_size//2),
                nn.ReLU(),
                nn.LayerNorm(hidden_size//2, eps=1e-12),
                nn.Dropout(dropout),
                nn.Linear(hidden_size//2, output_size)
            )

class MA2TransNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # text encoder
        txt_bert_config = BertConfig(self.args.hidden_size, self.args.num_hidden_layers,
                                     self.args.num_attention_heads, type_vocab_size=2)
        self.txt_encoder = BertModel.from_pretrained('pertrained/bert-base-uncased/', config=txt_bert_config)
        # object encoder
        self.obj_encoder = ObjEncoder(self.args.hidden_size)
        # attributes encoder
        self.attr_encoder = Attributes(self.args.hidden_size, self.args.num_hidden_layers)

        at_attention_layer = A_T_DecoderLayer(self.args.hidden_size, self.args.num_hidden_layers, dim_feedforward=2048, dropout=0.1)
        self.ATlayer = _get_clones(at_attention_layer, args.num_attlayers)

        av_attention_layer = A_V_DecoderLayer(self.args.hidden_size, self.args.num_hidden_layers, dim_feedforward=2048, dropout=0.1)
        self.ATlayer = _get_clones(av_attention_layer, args.num_attlayers)

        self.exchanginglayer = CrossExchangingLayer(self.args.hidden_size, args.nhead, args.theta, skip_connection=args.skip_connection,
                                                    use_quantile=args.use_quantile, dropout=args.cross_dropout)
        self.exhcangingblock = CrossExchangingBlock(encoder_layer=self.exchanginglayer, num_layers=args.num_exlayers,
                                                    replace_start=args.replace_start, replace_end=args.replace_end, norm=None)

        self.og3d_head = get_mlp_head(
            self.args.hidden_size, self.args.hidden_size,
            1, dropout=self.args.dropout)
        self.obj3d_clf_head = get_mlp_head(
            self.args.hidden_size, self.args.hidden_size,
            self.args.num_obj_classes, dropout=self.args.dropout)
        self.obj3d_clf_pre_head = get_mlp_head(
            self.args.hidden_size, self.args.hidden_size,
            self.args.num_obj_classes, dropout=self.args.dropout)
        self.obj3d_reg_head = get_mlp_head(
            self.args.hidden_size, self.args.hidden_size,
            3, dropout=self.args.dropout)
        self.txt_clf_head = get_mlp_head(
            self.args.hidden_size, self.args.hidden_size,
            self.args.num_obj_classes, dropout=self.args.dropout)

        self.apply(self._init_weights)

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch: dict)-> dict:
        batch = self.prepare_batch(batch)
        textout_embeds = self.txt_encoder(
            batch['txt_ids'], batch['txt_masks'],
        ).last_hidden_state
        objout_embeds = self.obj_encoder(batch['obj_fts'])
        obj_attrs = self.attr_encoder(batch['obj_colors'],batch['obj_locs'])
        txt_masks = batch['txt_masks']
        obj_masks = batch['obj_masks']
        for i, ATlayers in enumerate(self.ATlayer):
            textout_embeds, at_attn = ATlayers(
                textout_embeds, obj_attrs,
                tgt_key_padding_mask=obj_masks.logical_not(),
                memory_key_padding_mask=txt_masks.logical_not(),)
        for i, AVlayers in enumerate(self.AVlayer):
            objout_embeds, av_self_attn, av_cross_attn = AVlayers(
                objout_embeds, obj_attrs,
                tgt_key_padding_mask=obj_masks.logical_not(),
                memory_key_padding_mask=txt_masks.logical_not(),)
        av_cross_attn = torch.sigmoid(av_cross_attn)
        av_attn = torch.softmax(torch.log(torch.clamp(av_cross_attn, min=1e-6)) + av_self_attn)

        agg_feats1, agg_feats2 = self.exhcangingblock(textout_embeds, objout_embeds, at_attn, av_attn)
        og3d_logits = self.og3d_head(agg_feats1).squeeze(-1)

        result = {'og3d_logits': og3d_logits, }
        result['obj3d_clf_logits'] = self.obj3d_clf_head(objout_embeds)
        result['obj3d_loc_preds'] = self.obj3d_reg_head(objout_embeds)
        result['obj3d_clf_pre_logits'] = self.obj3d_clf_pre_head(objout_embeds)
        result['txt_clf_logits'] = self.txt_clf_head(textout_embeds[:, 0])

        # counterfactual attention
        fake_at_attn = torch.zeros_like(at_attn).uniform_(0, 2)
        fake_av_attn = torch.zeros_like(av_attn).uniform_(0, 2)
        fake_at_agg_feats1, fake_at_agg_feats2 = self.exhcangingblock(textout_embeds, objout_embeds, fake_at_attn, av_attn)
        fake_av_agg_feats1, fake_av_agg_feats2 = self.exhcangingblock(textout_embeds, objout_embeds, at_attn, fake_av_attn)
        result['fake_at_og3d_logits'] = self.og3d_head(fake_at_agg_feats1).squeeze(-1)
        result['fake_av_og3d_logits'] = self.og3d_head(fake_av_agg_feats1).squeeze(-1)

        # compute loss
        losses = self.compute_loss()
        return og3d_logits, losses

    def compute_loss(self, result, batch):
        losses = {}
        total_loss = 0

        og3d_loss = F.cross_entropy(result['og3d_logits'], batch['tgt_obj_idxs'])
        losses['og3d'] = og3d_loss
        total_loss += og3d_loss

        counterfactual_loss = 1/2 (F.cross_entropy(result['fake_at_og3d_logits'], batch['tgt_obj_idxs'])
                              + F.cross_entropy(result['fake_av_og3d_logits'], batch['tgt_obj_idxs']))
        losses['cf3d'] = counterfactual_loss
        total_loss += 0.5 * counterfactual_loss

        if self.args.obj3d_clf_loss > 0:
            obj3d_clf_loss = F.cross_entropy(
                result['obj3d_clf_logits'].permute(0, 2, 1),
                batch['obj_classes'], reduction='none'
            )
            obj3d_clf_loss = (obj3d_clf_loss * batch['obj_masks']).sum() / batch['obj_masks'].sum()
            losses['obj3d_clf'] = obj3d_clf_loss * self.args.obj3d_clf_loss
            total_loss += losses['obj3d_clf']

        if self.args.obj3d_clf_pre_loss > 0:
            obj3d_clf_pre_loss = F.cross_entropy(
                result['obj3d_clf_pre_logits'].permute(0, 2, 1),
                batch['obj_classes'], reduction='none'
            )
            obj3d_clf_pre_loss = (obj3d_clf_pre_loss * batch['obj_masks']).sum() / batch['obj_masks'].sum()
            losses['obj3d_clf_pre'] = obj3d_clf_pre_loss * self.args.obj3d_clf_pre_loss
            total_loss += losses['obj3d_clf_pre']

        if self.args.obj3d_reg_loss > 0:
            obj3d_reg_loss = F.mse_loss(
                result['obj3d_loc_preds'], batch['obj_locs'][:, :, :3],  reduction='none'
            )
            obj3d_reg_loss = (obj3d_reg_loss * batch['obj_masks'].unsqueeze(2)).sum() / batch['obj_masks'].sum()
            losses['obj3d_reg'] = obj3d_reg_loss * self.args.obj3d_reg_loss
            total_loss += losses['obj3d_reg']

        if self.args.txt_clf_loss > 0:
            txt_clf_loss = F.cross_entropy(
                result['txt_clf_logits'], batch['tgt_obj_classes'],  reduction='mean'
            )
            losses['txt_clf'] = txt_clf_loss * self.args.txt_clf_loss
            total_loss += losses['txt_clf']

        losses['total'] = total_loss













































