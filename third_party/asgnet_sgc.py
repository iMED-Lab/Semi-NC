"""SGC reference implementation from the user-supplied ASGNet source.

Source files:
  - ASGNet-main/model/asgnet.py::Weighted_GAP
  - ASGNet-main/model/asgnet.py::Model.sp_center_iter

Only integration edits were made: the method is a standalone function and
hard-coded `.cuda()` allocations use the input tensor's device. The association,
distance, normalization, and centroid-update operations are unchanged.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, gaussian_filter


def place_seed_points_fixed(mask, count):
    """ASGNet MaskSLIC placement with Semi-NC's paper-fixed prototype count."""
    segments_x = np.zeros(count, dtype=np.int64)
    segments_y = np.zeros(count, dtype=np.int64)
    m_np = mask.detach().cpu().numpy()
    nz = np.nonzero(m_np)
    if len(nz[0]) == 0:
        return torch.empty(0, 2, dtype=torch.long)

    p = [np.min(nz[0]), np.min(nz[1])]
    pend = [np.max(nz[0]), np.max(nz[1])]
    m_np_roi = np.copy(m_np)[p[0] : pend[0] + 1, p[1] : pend[1] + 1]
    for i in range(count):
        dtrans = gaussian_filter(distance_transform_edt(m_np_roi), sigma=0.1)
        coords = np.nonzero(dtrans == np.max(dtrans))
        segments_x[i] = coords[0][0]
        segments_y[i] = coords[1][0]
        m_np_roi[segments_x[i], segments_y[i]] = False
        segments_x[i] += p[0]
        segments_y[i] += p[1]
    segments = np.concatenate(
        [segments_x[..., np.newaxis], segments_y[..., np.newaxis]], axis=1
    )
    return torch.from_numpy(segments)


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def sp_center_iter(supp_feat, supp_mask, sp_init_center, n_iter):
    """Reference superpixel centroid iteration; returns C x num_sp."""
    c_xy, num_sp = sp_init_center.size()
    _, h, w = supp_feat.size()
    h_coords = torch.arange(h, device=supp_feat.device).view(h, 1).contiguous().repeat(1, w).unsqueeze(0).float()
    w_coords = torch.arange(w, device=supp_feat.device).repeat(h, 1).unsqueeze(0).float()
    supp_feat = torch.cat([supp_feat, h_coords, w_coords], 0)
    supp_feat_roi = supp_feat[:, (supp_mask == 1).squeeze()]

    num_roi = supp_feat_roi.size(1)
    supp_feat_roi_rep = supp_feat_roi.unsqueeze(-1).repeat(1, 1, num_sp)
    sp_center = torch.zeros_like(sp_init_center, device=supp_feat.device)

    for i in range(n_iter):
        if i == 0:
            sp_center_rep = sp_init_center.unsqueeze(1).repeat(1, num_roi, 1)
        else:
            sp_center_rep = sp_center.unsqueeze(1).repeat(1, num_roi, 1)
        assert supp_feat_roi_rep.shape == sp_center_rep.shape
        dist = torch.pow(supp_feat_roi_rep - sp_center_rep, 2.0)
        feat_dist = dist[:-2, :, :].sum(0)
        spat_dist = dist[-2:, :, :].sum(0)
        total_dist = torch.pow(feat_dist + spat_dist / 100, 0.5)
        p2sp_assoc = torch.neg(total_dist).exp()
        p2sp_assoc = p2sp_assoc / p2sp_assoc.sum(0, keepdim=True)
        sp_center = supp_feat_roi_rep * p2sp_assoc.unsqueeze(0)
        sp_center = sp_center.sum(1)
    return sp_center[:-2, :]
