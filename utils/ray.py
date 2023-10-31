import torch
import numpy as np
from utils.math import *

EPS = 1e-8

def get_rays_np(scene_info, c2w, coords=None, use_viewdir=True, use_pixel_centers=True):
    H, W, K = scene_info['H'], scene_info['W'], scene_info['K']
    pixel_center = .5 if use_pixel_centers else 0.

    if coords is not None:
        j, i = coords[..., 0] + pixel_center, coords[..., 1] + pixel_center
        if len(coords.shape) == 3:
            c2w = c2w[:, None]
    else:
        i, j = np.meshgrid(np.arange(W, dtype=np.float32) + pixel_center,
                           np.arange(H, dtype=np.float32) + pixel_center, indexing='xy')

    dirs = np.stack([(i - K[0, 2]) / K[0, 0], (j - K[1, 2]) / K[1, 1], K[2, 2] * np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[..., :3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[..., :3, -1], np.shape(rays_d))

    if use_viewdir:
        rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + EPS)

    return rays_o, rays_d


class TwoSphere:
    def __init__(self, sphere_params):
        self.radius = sphere_params['sphere_radius']
        self.center = sphere_params['sphere_center']
        self.out_dim = 4

    def ray_plane_intersection(self, rays):
        """Compute intersection of the ray with a sphere with radius and center."""
        center = torch.Tensor(self.center).to(rays['origin'].device)
        center = torch.broadcast_to(center, rays['origin'].shape)

        # compute intersections
        L_co = center - rays['origin']
        t_co = (L_co * rays['direction']).sum(-1, keepdims=True)
        square_d = (L_co * L_co).sum(-1, keepdims=True) - t_co**2
        square_t_cp = self.radius**2 - square_d
        intersect_mask = (square_t_cp > 0).float()  # only two-intersection is valid

        t_cp = torch.sqrt(square_t_cp * intersect_mask + EPS)
        t0 = t_co - t_cp
        t1 = t_co + t_cp

        p0 = rays['origin'] + t0 * rays['direction']
        p1 = rays['origin'] + t1 * rays['direction']

        # centered at coordinate origin
        p0 -= center
        p1 -= center

        # convert to spherical coordinate
        st = coord2sph(p0, normalize=True)
        uv = coord2sph(p1, normalize=True)
        samples = torch.cat([st, uv], -1)

        hit_info = {'t0': t0}
        return samples, hit_info

    def ray2param(self, x):
        """Compute the twosphere representation."""
        rays_dir = x[..., 3:6]
        rays_d_sph = coord2sph(rays_dir).requires_grad_(True)
        rays_d = sph2coord(rays_d_sph[..., 0], rays_d_sph[..., 1])
        rays = {
            'origin': x[..., :3],
            'direction': rays_d,  # differential ray direction
        }
        samples, hit_info = self.ray_plane_intersection(rays)

        hit_info['ray_dir'] = rays_d_sph  # return differential ray dir. to compute normals
        return samples, hit_info


def get_rayparam_func(scene_info):
    ray_param = TwoSphere(scene_info)
    ray_embed = lambda x, rp=ray_param: rp.ray2param(x)
    return ray_embed, ray_param.out_dim

def get_ray_param(ray_fn, rays):
    samples, hit_info = ray_fn(rays)
    return samples, hit_info['t0'].detach(), hit_info['ray_dir']
