import torch

EPS = 1e-8


def get_multiview_rays(args, query_rays, query_gts):
    # define the query surface points
    wcoords = query_rays[..., :3] + query_gts['dist'] * query_rays[..., 3:]

    # sample points on a unit sphere to construct vectors
    x = 2. * torch.rand([wcoords.shape[0], args.N_views]) - 1.
    y = 2. * torch.rand([wcoords.shape[0], args.N_views]) - 1.
    z = 2. * torch.rand([wcoords.shape[0], args.N_views]) - 1.
    mv_dirs = torch.stack([x, y, z], dim=-1).to(wcoords.device)
    mv_dirs = mv_dirs / (torch.linalg.norm(mv_dirs, dim=-1, keepdim=True) + EPS)
    rays_d = -mv_dirs

    # generate new rays
    dist = args.radius * 2.
    rays_o = wcoords[:, None] - dist * rays_d
    mv_rays = torch.concat([rays_o, rays_d], dim=-1)  # (B, N_views, 6)
    target_dict = {'dist': torch.ones_like(rays_d[..., :1]) * dist}
    if args.rgb_layer > 0:
        target_dict['image'] = torch.tile(query_gts['image'][:, None], (1, args.N_views, 1))

    mv_rays_flat = mv_rays.reshape(-1, 6)
    for k in target_dict:
        target_dict[k] = target_dict[k].reshape(-1, target_dict[k].shape[-1])

    return mv_rays_flat, target_dict
