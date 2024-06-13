import copy
import torch
import torch.nn as nn
import einops
from torch import nn, Tensor
from lib.pointnet2.pointnet2_modules import PointnetSAModule

def break_up_pc(pc: Tensor) -> [Tensor, Tensor]:
    """
    Split the pointcloud into xyz positions and features tensors.
    This method is taken from VoteNet codebase (https://github.com/facebookresearch/votenet)

    @param pc: pointcloud [N, 3 + C]
    :return: the xyz tensor and the feature tensor
    """
    xyz = pc[..., 0:3].contiguous()
    features = (
        pc[..., 3:].transpose(1, 2).contiguous()
        if pc.size(-1) > 3 else None
    )
    return xyz, features

class PointNetPP(nn.Module):
    """
    Pointnet++ encoder.
    For the hyper parameters please advise the paper (https://arxiv.org/abs/1706.02413)
    """
    def __init__(self, sa_n_points: list,
                 sa_n_samples: list,
                 sa_radii: list,
                 sa_mlps: list,
                 bn=True,
                 use_xyz=True):
        super().__init__()

        n_sa = len(sa_n_points)
        if not (n_sa == len(sa_n_samples) == len(sa_radii) == len(sa_mlps)):
            raise ValueError('Lens of given hyper-params are not compatible')

        self.encoder = nn.ModuleList()

        for i in range(n_sa):
            self.encoder.append(PointnetSAModule(
                npoint=sa_n_points[i],
                nsample=sa_n_samples[i],
                radius=sa_radii[i],
                mlp=sa_mlps[i],
                bn=bn,
                use_xyz=use_xyz,
            ))

        out_n_points = sa_n_points[-1] if sa_n_points[-1] is not None else 1
        self.fc = nn.Linear(out_n_points * sa_mlps[-1][-1], sa_mlps[-1][-1])

    def forward(self, features):
        """
        @param features: B x N_objects x N_Points x 3 + C
        """
        xyz, features = break_up_pc(features)
        for i in range(len(self.encoder)):
            xyz, features = self.encoder[i](xyz, features)

        return self.fc(features.view(features.size(0), -1))

class ObjEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # self.pcd_net = PointNetPP(
        #     sa_n_points=config.sa_n_points,
        #     sa_n_samples=config.sa_n_samples,
        #     sa_radii=config.sa_radii,
        #     sa_mlps=config.sa_mlps,
        # )
        self.object_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                         sa_n_samples=[[32], [32], [None]],
                                         sa_radii=[[0.2], [0.4], [None]],
                                         sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]])
        self.dropout = nn.Dropout(0.3)

    def forward(self, obj_pcds):
        batch_size, num_objs, _, _ = obj_pcds.size()
        obj_embeds = []
        for i in range(batch_size):
            obj_embeds.append(self.pcd_net(obj_pcds[i]))
        obj_embeds = torch.stack(obj_embeds, 0)

        obj_embeds = self.dropout(obj_embeds)
        return obj_embeds