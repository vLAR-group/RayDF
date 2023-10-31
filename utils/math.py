import numpy as np
import torch

EPS = 1e-8


def coord2sph(xyz, normalize=False, return_radius=False):
    xy = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    theta = torch.arctan2(torch.sqrt(xy), xyz[..., 2] + EPS)  # [0, pi]
    phi = torch.arctan2(xyz[..., 1], xyz[..., 0] + EPS)  # [-pi, pi]
    if normalize:  # normalize to [-1, 1]
        theta = 2. * (theta / torch.pi) - 1.
        phi = phi / torch.pi
    if not return_radius:
        return torch.stack([theta, phi], -1)
    else:
        rad = torch.sqrt((xyz**2).sum(-1))
        return torch.stack([theta, phi, rad], -1)


def sph2coord(theta, phi, r=1.):
    coord = torch.stack([torch.sin(theta) * torch.cos(phi) * r,
                         torch.sin(theta) * torch.sin(phi) * r,
                         torch.cos(theta) * r], dim=-1)
    return coord


def get_coord_from_pts(wcoords, w2c, scene_info):
    K = scene_info['K']
    pts = torch.cat([wcoords, torch.ones_like(wcoords[..., :1])], -1)  # (B, 4)
    coords = (w2c[None] @ pts[:, None, :, None]).squeeze(-1)  # (B, N, 4)
    x = (K[0, 2] + (coords[..., :1] * K[0, 0]) / (coords[..., 2:3] * K[2, 2] + EPS)).long().squeeze(-1)
    y = (K[1, 2] + (coords[..., 1:2] * K[1, 1]) / (coords[..., 2:3] * K[2, 2] + EPS)).long().squeeze(-1)
    return x, y

def convert_d(d, scene_info, out='dist'):
    H, W, focal = scene_info['H'], scene_info['W'], scene_info['focal']
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    L = np.sqrt(np.power(j - H / 2., 2) + np.power(i - W / 2., 2) + focal ** 2)
    fl = focal / L
    if out == 'dist':
        return d / fl
    elif out == 'dep':
        return d * fl
    else:
        raise NotImplementedError


def gradient(y, x, grad_outputs=None, create_graph=True):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=create_graph)[0]
    return grad


def get_surface_gradient(t, raydirs):
    dt = gradient(t, raydirs)
    return torch.norm(dt, dim=-1, keepdim=True)


def get_surface_normal(t, raydirs):
    dt = gradient(t, raydirs)
    dtdtheta, dtdphi = dt[..., :1], dt[..., 1:]
    sin_theta, cos_theta = torch.sin(raydirs[..., :1]), torch.cos(raydirs[..., :1])
    sin_phi, cos_phi = torch.sin(raydirs[..., 1:]), torch.cos(raydirs[..., 1:])
    dtheta = torch.cat([(dtdtheta * sin_theta + t * cos_theta) * cos_phi,
                        (dtdtheta * sin_theta + t * cos_theta) * sin_phi,
                        dtdtheta * cos_theta - t * sin_theta], dim=-1)
    dphi = torch.cat([(dtdphi * cos_phi - t * sin_phi) * sin_theta,
                      (dtdphi * sin_phi + t * cos_phi) * sin_theta,
                      dtdphi * cos_theta], dim=-1)

    normal = torch.cross(dphi, dtheta)
    normal = normal / (torch.linalg.norm(normal+EPS, dim=-1, keepdim=True)+EPS)
    return normal
