import einops
import torch
from torch import nn, Tensor
import config

class Attributes(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0):
        super().__init__()
        self.ft_linear = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Dropout(dropout)
        )
        self.project = nn.Linear(input_size, hidden_size)

    def forward(self, obj_colors, obj_locs):
        gmm_weights = obj_colors[..., :1]
        gmm_means = obj_colors[..., 1:]
        obj_colors = torch.sum(self.ft_linear(gmm_means) * gmm_weights, 2)
        pairwise_locs = self.calc_pairwise_locs(obj_locs[:, :, :3], obj_locs[:, :, 3:], pairwise_rel_type='center')
        attributes = torch.cat(obj_colors, pairwise_locs, dim=-1)
        attributes = self.project(attributes, config.hidden_size)
        return attributes

    def calc_pairwise_locs(self, obj_centers, obj_whls, eps=1e-10, pairwise_rel_type='center'):
        if pairwise_rel_type == 'mlp':
            obj_locs = torch.cat([obj_centers, obj_whls], 2)
            pairwise_locs = torch.cat(
                [einops.repeat(obj_locs, 'b l d -> b l x d', x=obj_locs.size(1)),
                 einops.repeat(obj_locs, 'b l d -> b x l d', x=obj_locs.size(1))],
                dim=3
            )
            return pairwise_locs

        pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
                        - einops.repeat(obj_centers, 'b l d -> b 1 l d')
        pairwise_dists = torch.sqrt(torch.sum(pairwise_locs ** 2, 3) + eps)  # (b, l, l)
        if self.config.spatial_dist_norm:
            max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
            norm_pairwise_dists = pairwise_dists / einops.repeat(max_dists, 'b -> b 1 1')
        else:
            norm_pairwise_dists = pairwise_dists

        if self.config.spatial_dim == 1:
            return norm_pairwise_dists.unsqueeze(3)

        pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, 3) + eps)
        if pairwise_rel_type == 'center':
            pairwise_locs = torch.stack(
                [norm_pairwise_dists, pairwise_locs[..., 2] / pairwise_dists,
                 pairwise_dists_2d / pairwise_dists, pairwise_locs[..., 1] / pairwise_dists_2d,
                 pairwise_locs[..., 0] / pairwise_dists_2d],
                dim=3
            )
        elif pairwise_rel_type == 'vertical_bottom':
            bottom_centers = torch.clone(obj_centers)
            bottom_centers[:, :, 2] -= obj_whls[:, :, 2]
            bottom_pairwise_locs = einops.repeat(bottom_centers, 'b l d -> b l 1 d') \
                                   - einops.repeat(bottom_centers, 'b l d -> b 1 l d')
            bottom_pairwise_dists = torch.sqrt(torch.sum(bottom_pairwise_locs ** 2, 3) + eps)  # (b, l, l)
            bottom_pairwise_dists_2d = torch.sqrt(torch.sum(bottom_pairwise_locs[..., :2] ** 2, 3) + eps)
            pairwise_locs = torch.stack(
                [norm_pairwise_dists,
                 bottom_pairwise_locs[..., 2] / bottom_pairwise_dists,
                 bottom_pairwise_dists_2d / bottom_pairwise_dists,
                 pairwise_locs[..., 1] / pairwise_dists_2d,
                 pairwise_locs[..., 0] / pairwise_dists_2d],
                dim=3
            )

        if self.config.spatial_dim == 4:
            pairwise_locs = pairwise_locs[..., 1:]
        return pairwise_locs