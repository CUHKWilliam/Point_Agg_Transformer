import torch


class Correlation:

    @classmethod
    def multilayer_correlation(cls, support_feats, query_feats):
        eps = 1e-5
        (
            query_batch_idxs,
            query_coords,
            query_locs_float,
            query_mask_features,
        ) = query_feats
        (
            support_batch_idxs,
            support_coords,
            support_locs_float,
            support_mask_features,
        ) = support_feats
        batch_size = torch.max(support_batch_idxs) + 1
        corr_coords = []
        corrs = []
        corr_batch_idxs = []
        corr_coord_floats = []
        for batch_id in range(batch_size):
            support_batch_mask, query_batch_mask = support_batch_idxs == batch_id, query_batch_idxs == batch_id
            support_batch_idxs_b, support_coords_b, support_locs_float_b, support_mask_features_b = \
                support_batch_idxs[support_batch_mask], support_coords[support_batch_mask], \
                support_locs_float[support_batch_mask], support_mask_features[support_batch_mask]
            query_batch_idxs_b, query_coords_b, query_locs_float_b, query_mask_features_b = \
                query_batch_idxs[query_batch_mask], query_coords[query_batch_mask], \
                query_locs_float[query_batch_mask], query_mask_features[query_batch_mask]

            # query x support

            corr_coord = torch.cat((support_coords_b.unsqueeze(0).repeat((query_coords_b.size(0), 1, 1)).reshape(-1, 3),
                       query_coords_b.unsqueeze(1).repeat((1, support_coords_b.size(0), 1)).reshape(-1, 3)), dim=-1)
            corr_coords.append(corr_coord)
            corr_coord_float = torch.cat((support_locs_float_b.unsqueeze(0).repeat((query_locs_float_b.size(0), 1, 1)).reshape(-1, 3),
                       query_locs_float_b.unsqueeze(1).repeat((1, support_locs_float_b.size(0), 1)).reshape(-1, 3)), dim=-1)
            corr_coord_floats.append(corr_coord_float)
            corr = ((support_mask_features_b / support_mask_features_b.norm(dim=1, p=2, keepdim=True)).unsqueeze(0) *
                    (query_mask_features_b / query_mask_features_b.norm(dim=1, p=2, keepdim=True)).unsqueeze(1)).sum(-1).reshape(-1, 1)

            corrs.append(corr)
            corr_batch_idx = torch.ones(len(corr)).to(corr.device) * batch_id
            corr_batch_idxs.append(corr_batch_idx)
        corr_coords = torch.cat(corr_coords, dim=0)
        corr_coords_float = torch.cat(corr_coord_floats, dim=0)
        corr_batch_idxs = torch.cat(corr_batch_idxs, dim=0)
        corrs = torch.cat(corrs, dim=0)
        return (
            corr_batch_idxs,
            corr_coords,
            corr_coords_float,
            corrs,
        )