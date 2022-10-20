import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base.encoder import MinkowskiEncoder, MinkowskiUpsample
import MinkowskiEngine as ME
from model.transformer_detr import CrossAttention
from MinkowskiEngine.utils.coords import get_coords_map
from model.pos_embedding import PositionEmbeddingCoordsSine
from util.visualize import visualize_pts_open3d

class MinskiUNet(ME.MinkowskiNetwork):
    def __init__(self, dim, c_in):
        super().__init__(dim)
        self.dim = dim
        self.encoder_corr1 = MinkowskiEncoder(
            dim=dim,
            channels=(c_in, 16, 16, 32, 32),
            kernel_size=(
                (3, 3, 3, 20, 1, 1),
                (3, 3, 3, 20, 1, 1),
                (3, 3, 3, 1, 20, 1),
                (3, 3, 3, 1, 20, 1),
            ),
            stride=(
                (2, 2, 2, 10, 1, 1),
                (1, 1, 1, 10, 1, 1),
                (2, 2, 2, 1, 10, 1),
                (1, 1, 1, 1, 10, 1),
            ),
            residual=False
            )
        self.encoder_corr2 = MinkowskiEncoder(
            dim=dim,
            channels=(64, 64, 64),
            kernel_size=(
                (3, 3, 3, 1, 1, 20),
                (3, 3, 3, 1, 1, 20),
            ),
            stride=(
                (2, 2, 2, 1, 1, 10),
                (1, 1, 1, 1, 1, 10),
            ),
            residual=False
        )
        self.encoder_q = MinkowskiEncoder(
            dim=3,
            channels=(16, 16, 16, 32, 32),
            kernel_size=(
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
            ),
            stride=(
                (2, 2, 2),
                (1, 1, 1),
                (2, 2, 2),
                (1, 1, 1),
            ),
            residual=False
            )

    def forward(self, corr, query):
        (
            corr_batch_idxs,
            corr_coords,
            corr_coords_float,
            corrs,
        ) = corr

        corr_in = ME.SparseTensor(corrs, coordinates=torch.cat((corr_batch_idxs.unsqueeze(-1), corr_coords), dim=-1))
        (
            query_batch_idxs,
            query_coords,
            query_coords_float,
            query_feats,
        ) = query
        query_in = ME.SparseTensor(query_feats,
                                   coordinates=torch.cat((query_batch_idxs.unsqueeze(-1), query_coords), dim=-1)
                                   )

        corr_out1 = self.encoder_corr1(corr_in)[0]
        query_out = self.encoder_q(query_in)[0]


        matched = (query_out.coordinates.unsqueeze(0) == corr_out1.coordinates[:, :4].unsqueeze(1)).all(-1)
        indices = torch.where(matched)[1]
        matched_query_feat = ME.SparseTensor(query_out.features[indices],
                                             coordinate_manager=corr_out1.coordinate_manager,
                                             coordinate_map_key=corr_out1.coordinate_map_key
                                             )
        corr_out2 = ME.cat(corr_out1, matched_query_feat)

        corr_out = self.encoder_corr2(corr_out2)[0]
        batch_idxs_out = corr_out.coordinates[:, 0]
        feature_out = corr_out.features.clone()
        corr_coords_float_out = []
        for b in range(batch_idxs_out.max() + 1):
            coords_supp_b = corr_out.coordinates[batch_idxs_out == b]
            feat_b = feature_out[batch_idxs_out == b]
            for coords_unique in torch.unique(coords_supp_b[:, -3:], dim=0):
                mask = torch.where((coords_supp_b[:, -3:] == coords_unique).all(-1))[0]
                feat_b[mask] = feat_b[mask].mean(0)
            feature_out[batch_idxs_out == b] = feat_b

            batch_mask = corr_batch_idxs == b
            offsets = corr_coords_float[batch_mask][:, : 3].min(0)[0]
            corr_coords_float_b = corr_coords_float[batch_mask][:, : 3] - offsets
            batch_mask_o = batch_idxs_out == b
            corr_coords_float_out_b = corr_out.coordinates[batch_mask_o][:, 1: 4] / \
                                      corr_coords[batch_mask][:, : 3].max(0)[0] * \
                                      corr_coords_float_b.max(0)[0]
            corr_coords_float_out_b += offsets
            corr_coords_float_out.append(corr_coords_float_out_b)
        corr_coords_float_out = torch.cat(corr_coords_float_out, dim=0)

        # for vis
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        #
        # pcd.points = o3d.utility.Vector3dVector(torch.cat([corr_coords_float[:, :3][corr_batch_idxs == 0],
        #                                                    corr_coords_float_out[corr_out.coordinates[:, 0] == 0]], dim=0).detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(torch.cat([torch.ones((len(corr_coords_float[corr_batch_idxs == 0]), 3)).float().cuda(),
        #                                                    torch.ones((len(corr_coords_float_out[corr_out.coordinates[:, 0] == 0]), 3)).float().cuda() *
        #                                                    torch.tensor([1, 0, 0]).float().cuda()], dim=0).detach().cpu().numpy())
        #
        # o3d.io.write_point_cloud("debug.ply", pcd)

        return (
            corr_out.coordinates[:, 0],
            corr_out.coordinates[:, 1:4],
            corr_coords_float_out,
            feature_out,
        )


class GuidedUpsample(nn.Module):
    def __init__(self, dim, dec_dim):
        super().__init__()
        self.cross_attention = CrossAttention(dim)
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=dec_dim, pos_type="fourier", normalize=True)

    def forward(self, corr, query_feats, pc_dims):
        (
            query_batch_idxs,
            query_coords,
            query_coords_float,
            querys,
        ) = query_feats
        (
            corr_batch_idxs,
            corr_coords,
            corr_coords_float,
            corrs,
        ) = corr


        relative_pos_batch = []
        corr_pos_batch = []
        query_pos_batch = []
        corr_batch = querys
        query_batch = corrs
        corr_mask_batch = []
        query_mask_batch = []

        for batch_id in range(query_batch_idxs.max() + 1):
            corr_batch_mask, query_batch_mask = corr_batch_idxs == batch_id, query_batch_idxs == batch_id
            corr_coords_b, query_coords_b = corr_coords[corr_batch_mask], query_coords[query_batch_mask]
            corr_coords_float_b, query_coords_float_b = corr_coords_float[corr_batch_mask], query_coords_float[query_batch_mask]
            relative_coords_b = query_coords_float_b[:, None, :] - corr_coords_float_b[None, :, :]
            n_queries, n_corrs = relative_coords_b.size(0), relative_coords_b.size(1)

            import ipdb;ipdb.set_trace()
            relative_emb_pos_b = self.pos_embedding(
                relative_coords_b.reshape(1, n_queries * n_corrs, -1), input_range=pc_dims
            ).reshape(
                -1,
                n_queries,
                n_corrs,
            )
            relative_pos_batch.append(relative_emb_pos_b)
            corr_emb_pos_b = self.pos_embedding(
                corr_coords_b, input_range=pc_dims
            )
            query_emb_pos_b = self.pos_embedding(
                query_coords_b, input_range=pc_dims
            )


        import ipdb;ipdb.set_trace()