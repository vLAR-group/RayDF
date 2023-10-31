import os
import torch
import torch.nn as nn

from utils.ray import get_rayparam_func
from utils.layer import Siren

EPS = 1e-8


class DualVisClassifier(nn.Module):
    def __init__(self, D=8, W=512, ext_layer=1, input_ch=11, w0_init=30.):
        super(DualVisClassifier, self).__init__()

        self.layer_ray = nn.ModuleList(
            [Siren(input_ch, W, w0=w0_init, is_first=True)] + [Siren(W, W) for i in range(ext_layer - 1)])
        self.layer_pts = nn.ModuleList(
            [Siren(3, W, w0=w0_init, is_first=True)] + [Siren(W, W) for i in range(ext_layer - 1)])
        self.lf_encoder = nn.ModuleList([Siren(W * 2, W)] + [Siren(W, W) for i in range(ext_layer + 1, D - 1)])
        self.cls_dense = Siren(W, 1, activation=nn.Identity())

    def forward(self, x):
        x_ray0, x_ray1, x_pts = x

        for i in range(len(self.layer_ray)):
            x_ray0 = self.layer_ray[i](x_ray0)
            x_ray1 = self.layer_ray[i](x_ray1)
            x_pts = self.layer_pts[i](x_pts)
        x_ray0_pts = torch.cat([x_ray0, x_pts], dim=-1)
        x_ray1_pts = torch.cat([x_ray1, x_pts], dim=-1)
        h = torch.stack([x_ray0_pts, x_ray1_pts], dim=-1).mean(-1)

        for i, l in enumerate(self.lf_encoder):
            h = self.lf_encoder[i](h)

        o = self.cls_dense(h)
        return o


def create_classifier(args, scene_info, device):
    ray_fn, input_ch = get_rayparam_func(scene_info)
    model = DualVisClassifier(D=args.netdepth_cls, W=args.netwidth_cls,
                              input_ch=input_ch, ext_layer=args.ext_layer_cls).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999), capturable=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_iters, eta_min=args.lrate * 0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lrate, total_steps=args.N_iters,
                                                    pct_start=0.3, three_phase=False)

    ############# Load checkpoints #############
    ckpts = [os.path.join(args.logdir, args.expname, f) for f in sorted(os.listdir(
        os.path.join(args.logdir, args.expname))) if 'tar' in f]
    print('Found ckpts', ckpts)

    start = 0
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print('Loading ckpt from:', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        model.load_state_dict(ckpt['network_fn'])
        optimizer.load_state_dict(ckpt['optimizer'])
        optimizer.param_groups[0]['capturable'] = True
        scheduler.load_state_dict(ckpt['scheduler'])
        scheduler.last_epoch = ckpt['global_step']

    return ray_fn, start, model, optimizer, scheduler
