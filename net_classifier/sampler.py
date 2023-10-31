import torch
from utils import math

EPS = 1e-8

def get_reference_rays(args, query_rays, query_dists, ref_dists, cams, scene_info):
    ref_dists = torch.tensor(ref_dists).to(query_dists.device)
    cams = torch.tensor(cams).to(query_dists.device)

    # define the query surface points
    query_pts = query_rays[..., :3] + query_dists * query_rays[..., 3:]

    # get all rays connecting new views with the query point
    cam_rays_o = torch.tile(cams[:, :3, -1][None], (query_rays.shape[0], 1, 1))
    cam_rays_d = query_pts[:, None] - cam_rays_o
    cam_rays_d = cam_rays_d / (torch.linalg.norm(cam_rays_d, dim=-1, keepdim=True) + EPS)
    cam_rays = torch.cat([cam_rays_o, cam_rays_d], dim=-1)

    # remove the query ray
    is_query = (query_rays[:, None, :3] == cam_rays_o).prod(-1)
    cam_rays = cam_rays[is_query == 0].reshape(query_rays.shape[0], -1, 6)

    # get the gt distance and fg mask of new rays
    ref_w2c = torch.linalg.pinv(cams)
    x, y = math.get_coord_from_pts(query_pts, ref_w2c, scene_info)
    ref_idx = torch.tile(torch.arange(len(ref_dists))[:, None], (1, x.shape[0]))  # (N, B)
    x = torch.clamp(x, 0, scene_info['W'] - 1).transpose(1, 0)  # (N, B)
    y = torch.clamp(y, 0, scene_info['H'] - 1).transpose(1, 0)
    gt_dists = ref_dists[ref_idx, y, x].transpose(1, 0)[is_query == 0].reshape(query_rays.shape[0], -1)

    # check if the query point is visible for ref views
    dists = torch.linalg.norm(query_pts[:, None] - cam_rays[..., :3], dim=-1)
    vis_mask = torch.isclose(dists, gt_dists, rtol=args.dist_thres).float()[..., None]

    return cam_rays, vis_mask
