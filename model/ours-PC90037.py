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

        _, unique_map, inverse_map = ME.utils.sparse_quantize(corr_coords.float(), return_index=True, return_inverse=True)
        corr_in = ME.SparseTensor(corrs[unique_map],
                                  coordinates=torch.cat((corr_batch_idxs[unique_map].unsqueeze(-1), corr_coords[unique_map]), dim=-1)
                                  )
        corr_coords_float_in = corr_coords_float[unique_map]
        (
            query_batch_idxs,
            query_coords,
            query_coords_float,
            query_feats,
        ) = query

        _, unique_map, inverse_map = ME.utils.sparse_quantize(query_coords.float(), return_index=True, return_inverse=True)

        query_in = ME.SparseTensor(query_feats[unique_map],
                                   coordinates=torch.cat((query_batch_idxs[unique_map].unsqueeze(-1), query_coords[unique_map]), dim=-1)
                                   )

        corr_out1 = self.encoder_corr1(corr_in)[0]
        query_out = self.encoder_q(query_in)[0]


        matched = []
        for b in range(torch.unique(query_out.coordinates[:, 0]).max() + 1):
            query_out_b = query_out.coordinates[query_out.coordinates[:, 0] == b]
            corr_out1_b = corr_out1.coordinates[corr_out1.coordinates[:, 0] == b]
            dist = ((query_out_b[:, 1:4].unsqueeze(0) - corr_out1_b[:, 1:4].unsqueeze(1)) ** 2).sum(-1)
            dist_min, matched_b = dist.min(-1)
            if b > 0:
                matched_b += (query_out.coordinates[:, 0] == b - 1).sum()
            matched.append(matched_b)
        matched = torch.cat(matched, dim=0)
        matched_query_feat = ME.SparseTensor(query_out.features[matched],
                                         coordinate_manager=corr_out1.coordinate_manager,
                                         coordinate_map_key=corr_out1.coordinate_map_key
                                         )

        corr_out2 = ME.cat(corr_out1, matched_query_feat)

        corr_out = self.encoder_corr2(corr_out2)[0]
        indices_in, indices_out = get_coords_map(corr_in, corr_out)
        indices_out_uniq = torch.unique(indices_out)
        coords_out_in_map = indices_out_uniq.unsqueeze(1) == indices_out.unsqueeze(0)
        corr_coords_float_out = [
            corr_coords_float_in[coords_out_in_map[i]].mean(0) for i in range(len(coords_out_in_map))
        ]

        corr_coords_float_out = torch.stack(corr_coords_float_out, dim=0)[:, :3]

        batch_idxs_out = corr_out.coordinates[:, 0]
        feature_out = corr_out.features.clone()
        for b in range(batch_idxs_out.max() + 1):
            coords_supp_b = corr_out.coordinates[batch_idxs_out == b]
            feat_b = feature_out[batch_idxs_out == b]
            for coords_unique in torch.unique(coords_supp_b[:, -3:], dim=0):
                mask = torch.where((coords_supp_b[:, -3:] == coords_unique).all(-1))[0]
                feat_b[mask] = feat_b[mask].mean(0)
            feature_out[batch_idxs_out == b] = feat_b

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